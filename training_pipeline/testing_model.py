from typing import Optional
import ndjson
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import pickle
import sys

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

from NodeToNodePaths import json_to_tree, find_tag_tree,  find_leaf_to_leaf_paths_iterative

if len(sys.argv) < 2:
    print("Usage: python AttentionCNNclassifier.py <fold_idx>")
    sys.exit(1)

fold_idx = int(sys.argv[1])  # Read and convert the fold index from command line argument


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
    func_root = json_to_tree(function_json)
    func_values, func_paths = find_leaf_to_leaf_paths_iterative(func_root)
    tag = find_tag_tree(func_root)

    _, func_paths = find_leaf_to_leaf_paths_iterative(func_root)  # get all contexts

    sts_indices = []  # start terminals indices
    paths_indices = []  # path indices
    ets_indices = []  # end terminals indices

    # Get the tag value, return None if not found (handle accordingly later)
    tag_idx = tags_vocab.get(tag, None)
    if tag_idx is None:
        return None  # handle appropriately

    for path in func_paths:  # map to the indices
        # Get the terminal node's data, if not in vocab skip adding to list
        start_index = value_vocab.get(path[0], None)
        if start_index is not None:
            sts_indices.append(start_index)

        # Get the path nodes' kinds, if not in vocab skip adding to list
        path_index = path_vocab.get(path[1:-1], None)
        if path_index is not None:
            paths_indices.append(path_index)

        # Get the ending terminal node's data, if not in vocab skip adding to list
        end_index = value_vocab.get(path[-1], None)
        if end_index is not None:
            ets_indices.append(end_index)

    # Pad sequences
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

total_processed = 0
successful_processed = 0
right_assigned = 0
top_5_right = 0

with open(test_file, 'r') as f:
    data = ndjson.load(f)

    for line in data:
        total_processed += 1

        try:
            tag_idx, sts_indices, value_indices, ets_indices = preprocess_function(
                line, value_vocab, path_vocab, tags_vocab, max_num_contexts)

            inputs = {
                'value1_input': np.array(value_indices),
                'path_input': np.array(sts_indices),
                'value2_input': np.array(ets_indices)
            }

            prediction = model.predict(inputs)

            # Get top 5 predictions
            top_5_indices = np.argsort(prediction[0])[-5:][::-1]  # Get top 5 indices in descending order
            top_5_probs = prediction[0][top_5_indices]  # Get the probabilities of the top 5 predictions

            predicted_function_tag = reverse_tags_vocab[np.argmax(prediction)]  # Using argmax for softmax output
            actual_tag = reverse_tags_vocab[tag_idx]

            if predicted_function_tag == actual_tag:
                print(f"CORRECT: Input function Tag: {actual_tag} | Predicted function Tag: {predicted_function_tag}")
                right_assigned += 1
            else:
                print(f"INCORRECT: Input function Tag: {actual_tag} | Predicted function Tag: {predicted_function_tag}")
                # Print top 5 predictions with their corresponding tag names and probabilities
                print("Top 5 Predictions:")
                for i in range(len(top_5_indices)):
                    predicted_tag_5 = reverse_tags_vocab[top_5_indices[i]]
                    if predicted_tag_5 == actual_tag:
                        top_5_right += 1
                    print(f"  Tag: {predicted_tag_5} | Probability: {top_5_probs[i]:.4f}")
            successful_processed += 1  # Increment the successful count if preprocessing is successful

        except Exception as e:
            print(f"Error processing line: {e}")

success_ratio = successful_processed / total_processed if total_processed > 0 else 0
right_ratio = right_assigned / total_processed if total_processed > 0 else 0

with open("analysis_csv/tests_results.log", "a") as log_file:
    success_ratio = successful_processed / total_processed if total_processed > 0 else 0
    right_ratio = right_assigned / total_processed if total_processed > 0 else 0
    incorrect_predictions = successful_processed - right_assigned
    top_5_ratio = (top_5_right / incorrect_predictions * 100) if incorrect_predictions > 0 else 0

    log_file.write(f"Fold {fold_idx}\n")
    log_file.write(f"Successfully processed: {successful_processed}/{total_processed} ({success_ratio * 100:.2f}%)\n")
    log_file.write(f"Correctly predicted from all: {right_assigned}/{total_processed} ({right_ratio * 100:.2f}%)\n")
    log_file.write(f"Correct prediction in top 5 from incorrectly predicted: {top_5_right}/{incorrect_predictions} ({top_5_ratio:.2f}%)\n")

# Print the same results to the console (optional)
print(f"Fold {fold_idx}")
print(f"Successfully processed: {successful_processed}/{total_processed} ({success_ratio * 100:.2f}%)")
print(f"Correctly predicted from all: {right_assigned}/{total_processed} ({right_ratio * 100:.2f}%)")
print(f"Correct prediction in top 5 from incorrectly predicted: {top_5_right}/{incorrect_predictions} ({top_5_ratio:.2f}%)")
