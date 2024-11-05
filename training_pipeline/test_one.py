#test git temp temp
from typing import Optional
import ndjson
import numpy as np
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

from NodeToNodePaths import json_to_tree, find_leaf_to_leaf_paths_iterative
from extract_functions.main import *

text_path = 'data_ndjson/one_func.txt'
ndjson_path = "data_ndjson/one_func.ndjson"
vocabs_pkl = 'vocabs.pkl'
model_file = 'model_fold_5.h5'
with open(ndjson_path, "w") as log_file:
    log_file.write("")

class WeightedContextLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightedContextLayer, self).__init__(**kwargs)

    def call(self, inputs):
        attention_weights, transformed_contexts = inputs
        weighted_context = tf.reduce_sum(attention_weights * transformed_contexts, axis=1)
        return weighted_context

class TagEmbeddingMatrixLayer(Layer):
    def __init__(self, tags_vocab_size, embedding_dim, **kwargs):
        super(TagEmbeddingMatrixLayer, self).__init__(**kwargs)
        self.tags_vocab_size = tags_vocab_size
        self.embedding_dim = embedding_dim
        self.tag_embedding = Embedding(input_dim=self.tags_vocab_size,
                                       output_dim=self.embedding_dim,
                                       name='tag_embedding',
                                       mask_zero=True)

    def call(self, inputs):
        tags_embedding_matrix = self.tag_embedding(tf.range(self.tag_embedding.input_dim))
        tags_embedding_matrix_t = tf.transpose(tags_embedding_matrix)

        num_repeats = tf.math.ceil((tf.shape(inputs)[1] / tf.shape(tags_embedding_matrix_t)[0]))
        num_repeats = tf.cast(num_repeats, tf.int32)

        tags_embedding_matrix_t_tiled = tf.tile(tags_embedding_matrix_t, [num_repeats, 1])
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
    func_values, func_paths = find_leaf_to_leaf_paths_iterative(func_root)

    sts_indices = []
    paths_indices = []
    ets_indices = []

    tag_idx = tags_vocab.get(tag, None)
    if tag_idx is None:
        return None

    for path in func_paths:
        start_index = value_vocab.get(path[0], None)
        if start_index is not None:
            sts_indices.append(start_index)

        path_index = path_vocab.get(path[1:-1], None)
        if path_index is not None:
            paths_indices.append(path_index)

        end_index = value_vocab.get(path[-1], None)
        if end_index is not None:
            ets_indices.append(end_index)

    sts_indices = pad_sequences([sts_indices], maxlen=max_num_contexts, padding='post', value=0)
    paths_indices = pad_sequences([paths_indices], maxlen=max_num_contexts, padding='post', value=0)
    ets_indices = pad_sequences([ets_indices], maxlen=max_num_contexts, padding='post', value=0)

    return tag_idx, sts_indices, paths_indices, ets_indices

def process_c_file(func_str: str, ndjson_path_t):
    print(func_str)
    prefix = "./extract_functions/"
    with open(f'{prefix}{temp_file_path}', 'w') as temp_file:
        # Write the cleaned content to the temp file
        temp_file.write(func_str)

    result = run_cnip(prefix) #------------------------------------------RUN--------------------

    # check exitcode, if error -> thrash the tree (don't save it)
    if result.returncode != 0:
        print(result.stderr)
        pass
    # if successful, process the ascii-tree
    else:
        ascii_to_ndjson(result.stdout, ndjson_path_t)
    return

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

with open(text_path, 'r') as file:
    func_str = file.read()

process_c_file(func_str, ndjson_path)
with open(ndjson_path, 'r') as f:
    data = ndjson.load(f)

    if len(data) == 0:
        print("No data found in one_func.ndjson")
        sys.exit(1)

    function_json = data[0]  # Take the first function if there are multiple entries

    try:
        tag_idx, sts_indices, value_indices, ets_indices = preprocess_function(
            function_json, value_vocab, path_vocab, tags_vocab, max_num_contexts)

        inputs = {
            'value1_input': np.array(value_indices),
            'path_input': np.array(sts_indices),
            'value2_input': np.array(ets_indices)
        }

        prediction = model.predict(inputs)

        # Get top 5 predictions
        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
        top_5_probs = prediction[0][top_5_indices]

        predicted_function_tag = reverse_tags_vocab[np.argmax(prediction)]
        actual_tag = reverse_tags_vocab[tag_idx]

        if predicted_function_tag == actual_tag:
            print(f"CORRECT: Input function Tag: {actual_tag} | Predicted function Tag: {predicted_function_tag}")
            print("Top 5 Predictions:")
            for i in range(len(top_5_indices)):
                predicted_tag_5 = reverse_tags_vocab[top_5_indices[i]]
                print(f"  Tag: {predicted_tag_5} | Probability: {top_5_probs[i]:.4f}")
        else:
            print(f"INCORRECT: Input function Tag: {actual_tag} | Predicted function Tag: {predicted_function_tag}")
            print("Top 5 Predictions:")
            for i in range(len(top_5_indices)):
                predicted_tag_5 = reverse_tags_vocab[top_5_indices[i]]
                print(f"  Tag: {predicted_tag_5} | Probability: {top_5_probs[i]:.4f}")

    except Exception as e:
        print(f"Error processing function: {e}")
