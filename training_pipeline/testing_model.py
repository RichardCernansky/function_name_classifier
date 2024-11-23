from typing import Optional
import ndjson
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import pickle
import sys
import pandas as pd
import json

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.saving import custom_object_scope
from keras.layers import Input, Embedding, Dense, Activation, Lambda, Concatenate, Softmax, Layer
from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.utils import pad_sequences, plot_model
from keras.activations import softmax
from collections import OrderedDict  # for ordered sets of the data
from sklearn.metrics import classification_report, accuracy_score

from NodeToNodePaths import json_to_tree, find_leaf_to_leaf_paths_iterative

if len(sys.argv) < 2:
    print("Usage: python AttentionCNNclassifier.py <fold_idx>")
    sys.exit(1)

fold_idx = int(sys.argv[1])

class WeightedContextLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightedContextLayer, self).__init__(**kwargs)

    def call(self, inputs):
        attention_weights, transformed_contexts = inputs
        # Compute the weighted context
        weighted_context = tf.reduce_sum(attention_weights * transformed_contexts, axis=1)
        return weighted_context

class TagEmbeddingMatrixLayer(Layer):
    def __init__(self, tags_vocab_size, embedding_dim, **kwargs):
        super(TagEmbeddingMatrixLayer, self).__init__(**kwargs)
        self.tags_vocab_size = tags_vocab_size
        self.embedding_dim = embedding_dim
        self.tag_embedding = None  # Initialize here
        self.tag_embedding = Embedding(input_dim=self.tags_vocab_size,
                                       output_dim=self.embedding_dim,
                                       name='tag_embedding',
                                       mask_zero=True)

    def call(self, inputs):
        # transpose the tag embeddings
        tags_embedding_matrix = self.tag_embedding(
            tf.range(self.tag_embedding.input_dim))  # Shape: (tags_vocab_size, embedding_dim)
        tags_embedding_matrix_t = tf.transpose(tags_embedding_matrix)  # Shape: (embedding_dim, tags_vocab_size)

        # num_repeats based on the shape of weighted_context
        num_repeats = tf.math.ceil((tf.shape(inputs)[1] / tf.shape(tags_embedding_matrix_t)[0]))
        num_repeats = tf.cast(num_repeats, tf.int32)  # Ensure it's an integer

        # # tile it
        tags_embedding_matrix_t_tiled = tf.tile(tags_embedding_matrix_t, [num_repeats,
                                                                          1])  # Shape: (num_repeats * embedding_dim, tags_vocab_size)

        # # only the required portion         # Shape: (weighted_context.shape[1], tags_vocab_size
        matrix_final = tf.matmul(inputs, tags_embedding_matrix_t_tiled[:(tf.shape(inputs)[1])])

        return matrix_final

def softmaxAxis1(x):
    return softmax(x, axis=1)

def get_vocabs(vocabs_pkl):
    with open(vocabs_pkl, 'rb') as f:
        vocabs = pickle.load(f)
        return vocabs['value_vocab'], vocabs['path_vocab'], vocabs['tags_vocab'], vocabs['max_num_contexts']

def preprocess_function(function_json, value_vocab, path_vocab, tags_vocab, max_num_contexts):
    tag = function_json.get('tag')
    func = function_json.get('ast')
    func_root = json_to_tree(func)

    _, func_paths = find_leaf_to_leaf_paths_iterative(func_root)  # get all contexts

    sts_indices = []  # start terminals indices
    paths_indices = []  # path indices
    ets_indices = []  # end terminals indices

    tag_idx = tags_vocab.get(tag, None)
    if tag_idx is None:
        return None  # handle appropriately

    for path in func_paths:  # map to the indices
        # get the terminal node's data, if not in vocab skip adding to list
        start_index = value_vocab.get(path[0], None)
        if start_index is not None:
            sts_indices.append(start_index)

        # get the path nodes' kinds, if not in vocab skip adding to list
        path_index = path_vocab.get(path[1:-1], None)
        if path_index is not None:
            paths_indices.append(path_index)

        # get the ending terminal node's data, if not in vocab skip adding to list
        end_index = value_vocab.get(path[-1], None)
        if end_index is not None:
            ets_indices.append(end_index)

    # pad sequences
    sts_indices = pad_sequences([sts_indices], maxlen=max_num_contexts, padding='post', value=0)
    paths_indices = pad_sequences([paths_indices], maxlen=max_num_contexts, padding='post', value=0)
    ets_indices = pad_sequences([ets_indices], maxlen=max_num_contexts, padding='post', value=0)

    return tag_idx, sts_indices, paths_indices, ets_indices

vocabs_pkl = 'vocabs.pkl'
test_file = 'data_ndjson/test_fold.ndjson'
model_file = f'model_fold_{fold_idx}.h5'

value_vocab, path_vocab, tags_vocab, max_num_contexts = get_vocabs(vocabs_pkl)

reverse_value_vocab = {idx: value for value, idx in value_vocab.items()}
reverse_path_vocab = {idx: path for path, idx in path_vocab.items()}
reverse_tags_vocab = {idx: tag for tag, idx in tags_vocab.items()}

custom_objects = {
    'softmaxAxis1': softmaxAxis1,
    'WeightedContextLayer': WeightedContextLayer,
    'TagEmbeddingMatrixLayer': TagEmbeddingMatrixLayer
}
with custom_object_scope(custom_objects):
    model = load_model(model_file)

true_labels = []
predicted_labels = []

bins = list(range(0, 2100, 100))  # bins from 0 to 2000 with a step of 100
bin_labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]

# dictionaries to keep track of correct predictions and total samples per bin
correct_predictions_per_bin = {label: 0 for label in bin_labels}
total_samples_per_bin = {label: 0 for label in bin_labels}

# process the test data and gather predictions and bin information
with open(test_file, 'r') as f:
    data = ndjson.load(f)

    for line in data:
        try:
            num_tokens = line.get("num_tokens")
            if num_tokens is None:
                continue  # skip if num_tokens is missing

            true_bin = pd.cut([num_tokens], bins=bins, labels=bin_labels)[0]

            result = preprocess_function(line, value_vocab, path_vocab, tags_vocab, max_num_contexts)
            if result is None:
                continue

            tag_idx, sts_indices, value_indices, ets_indices = result

            inputs = {
                'value1_input': np.array(value_indices),
                'path_input': np.array(sts_indices),
                'value2_input': np.array(ets_indices)
            }

            prediction = model.predict(inputs)
            predicted_tag_idx = np.argmax(prediction)

            # store true and predicted labels for the overall report
            true_labels.append(tag_idx)
            predicted_labels.append(predicted_tag_idx)

            # update bin-based accuracy counters
            if predicted_tag_idx == tag_idx:
                correct_predictions_per_bin[true_bin] += 1
            total_samples_per_bin[true_bin] += 1

        except Exception as e:
            print(f"Error processing line: {e}")

# calculate overall accuracy and classification report
accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels,
                               predicted_labels,
                               target_names=[reverse_tags_vocab[idx] for idx in set(true_labels)],
                               output_dict=True)

fold_metrics = {
    "accuracy": accuracy,
    "classification_report": report,
    "bin_accuracies": {
        bin_label: {
            "correct": correct_predictions_per_bin[bin_label],
            "total": total_samples_per_bin[bin_label]
        } for bin_label in bin_labels
    }
}

with open(f"analysis/fold_{fold_idx}_metrics.json", "w") as f:
    json.dump(fold_metrics, f, indent=4)

for bin_label in bin_labels:
    correct = correct_predictions_per_bin[bin_label]
    total = total_samples_per_bin[bin_label]
    accuracy_bin = correct / total if total > 0 else 0
    print(f"Bin {bin_label}: Accuracy = {accuracy_bin:.2f}")
print(f"Fold {fold_idx} Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{report}")

