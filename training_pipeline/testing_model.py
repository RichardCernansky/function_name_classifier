from typing import Optional
import ndjson
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import pickle

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

    tag_idx = tags_vocab[tag]  # get the tag value

    for path in func_paths:  # map to the indices
        sts_indices.append(value_vocab[path[0]])  # get the terminal node's data
        paths_indices.append(path_vocab[path[1:-1]])  # get the path nodes' kinds
        ets_indices.append(value_vocab[path[-1]])  # get the ending terminal node's data

    sts_indices = pad_sequences([sts_indices], maxlen=max_num_contexts, padding='post', value=0)
    paths_indices = pad_sequences([paths_indices], maxlen=max_num_contexts, padding='post', value=0)
    ets_indices = pad_sequences([ets_indices], maxlen=max_num_contexts, padding='post', value=0)

    return tag_idx, sts_indices, paths_indices, ets_indices


vocabs_pkl = 'vocabs.pkl'
test_file = 'data_ndjson/strat_test_functionsASTs.ndjson'
model_file = 'NEDELA_func_classifier_model.h5'

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

with open(test_file, 'r') as f:
    data = ndjson.load(f)

    for line in data:
        total_processed += 1

        try:
            tag_idx, sts_indices, value_indices, ets_indices = preprocess_function(line, value_vocab, path_vocab,
                                                                                   tags_vocab, max_num_contexts)
            print(tag_idx)

            inputs = {
                'value1_input': np.array(value_indices),
                'path_input': np.array(sts_indices),
                'value2_input': np.array(ets_indices)
            }

            prediction = model.predict(inputs)

            predicted_function_tag = reverse_tags_vocab[np.argmax(prediction)]  # Using argmax for softmax output
            actual_tag = reverse_tags_vocab[tag_idx]

            if predicted_function_tag == actual_tag:
                right_assigned += 1
            else:
                print(f"INCORRECT: Input function Tag: {actual_tag} | Predicted function Tag: {predicted_function_tag}")
            successful_processed += 1  # Increment the successful count if preprocessing is successful

        except Exception as e:
            print(f"Error processing line: {e}")

success_ratio = successful_processed / total_processed if total_processed > 0 else 0
right_ratio = right_assigned / successful_processed if total_processed > 0 else 0
print(f"Successfully processed: {successful_processed}/{total_processed} ({success_ratio * 100:.2f}%)")
print(
    f"Correctly predicted from successfully processed: {right_assigned}/{successful_processed} ({right_ratio * 100:.2f}%)")

