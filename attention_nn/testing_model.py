from typing import Optional
import ndjson
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import pickle
import random
import sys
import pandas as pd
import json
import os

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import custom_object_scope
from keras.layers import Input, Embedding, Dense, Activation, Lambda, Concatenate, Softmax, Layer
from keras.models import Model
from keras.callbacks import LambdaCallback
from keras.utils import pad_sequences, plot_model
from keras.activations import softmax
from collections import OrderedDict  # for ordered sets of the data
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from NodeToNodePaths import json_to_tree, find_leaf_to_leaf_paths_iterative

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from code_metrics import compute_halstead

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
    sampled_paths = random.sample(func_paths, min(max_num_contexts, len(func_paths)))

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

vocabs_pkl = f'trained_models/vocabs_fold_{fold_idx}.pkl'
test_file = 'data_ndjson/test_fold.ndjson'
model_file = f'trained_models/model_fold_{fold_idx}.h5'

value_vocab, path_vocab, tags_vocab, max_num_contexts = get_vocabs(vocabs_pkl)
max_num_contexts = 600


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

#------------------------------ACTUAL-TESTING--------------------------
true_labels = []
predicted_labels = []

def generate_bins_and_labels(start, end, step):
    bins = list(range(start, end + step, step))  # Add step to include the last bin edge
    bin_labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]

    correct_predictions_per_bin = {label: 0 for label in bin_labels}
    total_samples_per_bin = {label: 0 for label in bin_labels}
    return bins, bin_labels, correct_predictions_per_bin, total_samples_per_bin


num_tokens_bins_50, num_tokens_bin_labels_50, num_tokens_correct_50, num_tokens_total_50 = generate_bins_and_labels(0, 500, 50)
num_tokens_bins_20, num_tokens_bin_labels_20, num_tokens_correct_20, num_tokens_total_20 = generate_bins_and_labels(0, 500, 20)

ast_depth_bins_5, ast_depth_bin_labels_5, ast_depth_correct_5, ast_depth_total_5 = generate_bins_and_labels(0, 50, 5)
ast_depth_bins_2, ast_depth_bin_labels_2, ast_depth_correct_2, ast_depth_total_2 = generate_bins_and_labels(0, 50, 2)

num_nodes_bins_50, num_nodes_bin_labels_50, num_nodes_correct_50, num_nodes_total_50 = generate_bins_and_labels(0, 600, 50)
num_nodes_bins_20, num_nodes_bin_labels_20, num_nodes_correct_20, num_nodes_total_20 = generate_bins_and_labels(0, 600, 20)


lengths_tokens = []
source_codes = []
num_tokens_arr = []
# process the test data and gather predictions and bin information
with open(test_file, 'r') as f:
    data = ndjson.load(f)
    for line in data:
        try:
            num_tokens = line.get("num_tokens")
            ast_depth = line.get("ast_depth")
            num_nodes = line.get("num_nodes")
            source_code = line.get("source_code")
            source_codes.append(source_code)
            lengths_tokens.append(num_tokens)
            # lengths_tokens.append(compute_halstead(source_code)) #for halstead complexity
            num_tokens_arr.append(num_tokens)
            
            
            if any(value is None for value in [num_tokens, ast_depth, num_nodes]):
                continue  # skip if any of them is missing

            num_tokens_true_bin_50 = pd.cut([num_tokens], bins=num_tokens_bins_50, labels=num_tokens_bin_labels_50)[0]
            num_tokens_true_bin_20 = pd.cut([num_tokens], bins=num_tokens_bins_20, labels=num_tokens_bin_labels_20)[0]

            ast_depth_true_bin_5 = pd.cut([ast_depth], bins=ast_depth_bins_5, labels=ast_depth_bin_labels_5)[0]
            ast_depth_true_bin_2 = pd.cut([ast_depth], bins=ast_depth_bins_2, labels=ast_depth_bin_labels_2)[0]

            num_nodes_true_bin_50 = pd.cut([num_nodes], bins=num_tokens_bins_50, labels=num_tokens_bin_labels_50)[0]
            num_nodes_true_bin_20 = pd.cut([num_nodes], bins=num_tokens_bins_20, labels=num_tokens_bin_labels_20)[0]

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
                num_tokens_correct_50[num_tokens_true_bin_50] += 1
                num_tokens_correct_20[num_tokens_true_bin_20] += 1
                ast_depth_correct_5[ast_depth_true_bin_5] += 1
                ast_depth_correct_2[ast_depth_true_bin_2] += 1
                num_nodes_correct_50[num_nodes_true_bin_50] += 1
                num_nodes_correct_20[num_nodes_true_bin_20] += 1

            num_tokens_total_50[num_tokens_true_bin_50] += 1
            num_tokens_total_20[num_tokens_true_bin_20] += 1
            ast_depth_total_5[ast_depth_true_bin_5] += 1
            ast_depth_total_2[ast_depth_true_bin_2] += 1
            num_nodes_total_50[num_nodes_true_bin_50] += 1
            num_nodes_total_20[num_nodes_true_bin_20] += 1

        except Exception as e:
            print(f"Error processing line: {e}")


#calculate overall accuracy and classification report
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)


all_labels = sorted(reverse_tags_vocab.keys())
target_names = [reverse_tags_vocab[idx] for idx in all_labels]
report = classification_report(
    true_labels,
    predicted_labels,
    labels=all_labels,  # Explicitly specify all classes
    target_names=target_names,
    output_dict=True
)

#Confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
decoded_labels_sorted = [reverse_tags_vocab[i] for i in sorted(reverse_tags_vocab.keys()) if i != 0]
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="coolwarm", linewidths=0.5, xticklabels=decoded_labels_sorted, yticklabels=decoded_labels_sorted)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.savefig("conf_matrix.pdf", format="pdf", bbox_inches="tight")

fold_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "classification_report": report,
    "num_tokens_50_bin_accuracies": {
        bin_label: {
            "correct": num_tokens_correct_50[bin_label],
            "total": num_tokens_total_50[bin_label]
        } for bin_label in num_tokens_bin_labels_50
    },
    "num_tokens_20_bin_accuracies": {
        bin_label: {
            "correct": num_tokens_correct_20[bin_label],
            "total": num_tokens_total_20[bin_label]
        } for bin_label in num_tokens_bin_labels_20
    },
    "ast_depth_5_bin_accuracies": {
        bin_label: {
            "correct": ast_depth_correct_5[bin_label],
            "total": ast_depth_total_5[bin_label]
        } for bin_label in ast_depth_bin_labels_5
    },
    "ast_depth_2_bin_accuracies": {
        bin_label: {
            "correct": ast_depth_correct_2[bin_label],
            "total": ast_depth_total_2[bin_label]
        } for bin_label in ast_depth_bin_labels_2
    },
    "num_nodes_50_bin_accuracies": {
        bin_label: {
            "correct": num_nodes_correct_50[bin_label],
            "total": num_nodes_total_50[bin_label]
        } for bin_label in num_nodes_bin_labels_50
    },
    "num_nodes_20_bin_accuracies": {
        bin_label: {
            "correct": num_nodes_correct_20[bin_label],
            "total": num_nodes_total_20[bin_label]
        } for bin_label in num_nodes_bin_labels_20
    }
}

with open(f"analysis/metrics_json/fold_{fold_idx}_metrics.json", "w") as f:
    json.dump(fold_metrics, f, indent=4)

# for bin_label in bin_labels:
#     correct = correct_predictions_per_bin[bin_label]
#     total = total_samples_per_bin[bin_label]
#     accuracy_bin = correct / total if total > 0 else 0
#     print(f"Bin {bin_label}: Accuracy = {accuracy_bin:.2f}")
print(f"Fold {fold_idx} Results:")

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1: {f1:.4f}")

lengths_misclassified = []
for i in range(len(predicted_labels)):
    if predicted_labels[i] != true_labels[i]: lengths_misclassified.append(lengths_tokens[i])

sungyz_container = 0
sungyz_counter = 0
with open("data_ndjson/sungyz_predicted_wrong.txt", 'w') as f:
    for i in range(len(predicted_labels)):
        if reverse_tags_vocab[true_labels[i]] != "sungyz" and reverse_tags_vocab[predicted_labels[i]] == "sungyz":
            f.write(source_codes[i])
            sungyz_counter = sungyz_counter + num_tokens_arr[i]
            sungyz_container = sungyz_container + 1
print("Average sungyz token length: ", sungyz_counter/sungyz_container)   


ml_json_filename = "../misclass_lens.json"
with open(ml_json_filename) as f:
    file_dict = json.load(f)

file_dict["att-nn"] += lengths_misclassified

with open(ml_json_filename, "w") as f:
    json.dump(file_dict, f)
