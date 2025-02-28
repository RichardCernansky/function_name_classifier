from typing import Optional
import ndjson
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import time 
import pickle
import sys
import random

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Embedding, Dense, Activation, Lambda, Concatenate, Softmax, Layer
from keras.models import Model
from keras.callbacks import LambdaCallback,EarlyStopping, Callback
from keras.utils import pad_sequences, plot_model
from keras.activations import softmax
from collections import OrderedDict  # for ordered sets of the data

from extract_functions.Node import Node
from NodeToNodePaths import json_to_tree, find_leaf_to_leaf_paths_iterative

if len(sys.argv) < 2:
    print("Usage: python AttentionCNNclassifier.py <fold_idx>")
    sys.exit(1)

fold_idx = int(sys.argv[1])  # Read and convert the fold index from command line argument


def count_ndjson_lines(file_name):
    with open(file_name, 'r') as f:
        return sum(1 for line in f if line.strip())  # Count non-empty lines

train_file = "data_ndjson/strat_train.ndjson"
valid_file = "data_ndjson/strat_valid.ndjson"

number_lines_train = count_ndjson_lines(train_file)
number_lines_valid = count_ndjson_lines(valid_file)


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


# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------START-TRAINING--------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
print("Started generating vocabs...")


def get_vocabs(vocabs_pkl):
    with open(vocabs_pkl, 'rb') as f:
        vocabs = pickle.load(f)
        return vocabs['value_vocab'], vocabs['path_vocab'], vocabs['tags_vocab'], vocabs['max_num_contexts']


vocabs_pkl = f'trained_models/vocabs_fold_{fold_idx}.pkl'
value_vocab, path_vocab, tags_vocab, max_num_contexts = get_vocabs(vocabs_pkl)
# max_num_contexts = 600

# vocab sizes and embedding dimensions
value_vocab_size = len(value_vocab)
path_vocab_size = len(path_vocab)
tags_vocab_size = len(tags_vocab)
embedding_dim = 128
y = embedding_dim  # must be >= emb_dim
batch_size = 4  # number of functions trained at a time

print("--------------------DONE--------------------")
print(f"value_vocab_size: {value_vocab_size}")
print(f"path_vocab_size: {path_vocab_size}")
print(f"tags_vocab_size: {tags_vocab_size}")
print(f"max_num_contexts: {max_num_contexts}")

# inputs for value1, path, and value2 (with num_context inputs per batch)
input_value1 = Input(shape=(max_num_contexts,), name='value1_input')
input_path = Input(shape=(max_num_contexts,), name='path_input')
input_value2 = Input(shape=(max_num_contexts,), name='value2_input')

# embedding layers with mask_zero=True to handle padding (index 0)
value_embedding = Embedding(input_dim=value_vocab_size, output_dim=embedding_dim, name='value_embedding',
                            mask_zero=True)
path_embedding = Embedding(input_dim=path_vocab_size, output_dim=embedding_dim, name='path_embedding', mask_zero=True)

# embed the inputs
embedded_value1 = value_embedding(input_value1)  # shape: (None, num_context, embedding_dim)
embedded_path = path_embedding(input_path)  # shape: (None, num_context, embedding_dim)
embedded_value2 = value_embedding(input_value2)  # shape: (None, num_context, embedding_dim)

# concatenate along the last axis (for each context, value1, path, and value2 are concatenated)
embedded_concat = Concatenate(axis=-1)([embedded_value1, embedded_path, embedded_value2])
# Shape: (None, num_context, 3 * embedding_dim)

# apply a dense transformation to each concatenated context (row-wise transformation)
transformed_contexts = Dense(units=y, activation='tanh')(embedded_concat)
# Shape: (None, num_context, y)

# attention mechanism
attention_weights = Dense(1, activation=softmaxAxis1)(transformed_contexts)
# Shape: (None, num_context,1) - attention scores for each context

# apply attention weights to get the weighted sum of contexts
weighted_context = WeightedContextLayer()([attention_weights, transformed_contexts])
# shape: (None, y) - weighted sum across contexts

# get tags_embeddings transposed
tag_scores = TagEmbeddingMatrixLayer(tags_vocab_size, embedding_dim)(weighted_context)

output_tensor = Softmax()(tag_scores)

# define the model
model = Model(inputs=[input_value1, input_path, input_value2], outputs=output_tensor)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# train
print("\n\nStarted training on functions... for tag_size: ",tags_vocab_size)

total_loss = 0
batch_count = 0

def data_generator(functionsASTs_file, batch_size):
    while True:
        # Open the file and load all data for this epoch
        with open(functionsASTs_file, 'r') as ndjson_file:
            data = ndjson.load(ndjson_file)
        np.random.shuffle(data)
        
        batch_sts_indices = []
        batch_paths_indices = []
        batch_ets_indices = []
        batch_tag_index = []
        
        for function_json in data:
            tag = function_json.get('tag')
            func = function_json.get('ast')
            func_root = json_to_tree(func)
            _, func_paths = find_leaf_to_leaf_paths_iterative(func_root)  # get all contexts
            # sampled_paths = random.sample(func_paths, min(max_num_contexts, len(func_paths)))

            sts_indices = []  # start terminals indices
            paths_indices = []  # path indices
            ets_indices = []  # end terminals indices

            tag_index = tags_vocab[tag]  # get the tag value

            for path in func_paths:  # map each context to indices
                sts_indices.append(value_vocab[path[0]])
                paths_indices.append(path_vocab[path[1:-1]])
                ets_indices.append(value_vocab[path[-1]])

            # Pad sequences for consistency
            sts_indices = pad_sequences([sts_indices], maxlen=max_num_contexts, padding='post', value=0)
            paths_indices = pad_sequences([paths_indices], maxlen=max_num_contexts, padding='post', value=0)
            ets_indices = pad_sequences([ets_indices], maxlen=max_num_contexts, padding='post', value=0)

            batch_sts_indices.append(sts_indices)
            batch_paths_indices.append(paths_indices)
            batch_ets_indices.append(ets_indices)
            batch_tag_index.append(tag_index)

            # Yield a batch once enough examples have been accumulated
            if len(batch_sts_indices) == batch_size:
                sts_tensor = tf.convert_to_tensor(np.vstack(batch_sts_indices), dtype=tf.int32)
                paths_tensor = tf.convert_to_tensor(np.vstack(batch_paths_indices), dtype=tf.int32)
                ets_tensor = tf.convert_to_tensor(np.vstack(batch_ets_indices), dtype=tf.int32)
                tag_tensor = tf.convert_to_tensor(np.array(batch_tag_index, dtype=np.int64), dtype=tf.int64)

                yield (sts_tensor, paths_tensor, ets_tensor), tag_tensor

                # Reset batch lists for the next batch
                batch_sts_indices = []
                batch_paths_indices = []
                batch_ets_indices = []
                batch_tag_index = []

        # If there's a partial batch left over at the end, yield it as well
        if batch_sts_indices:
            sts_tensor = tf.convert_to_tensor(np.vstack(batch_sts_indices), dtype=tf.int32)
            paths_tensor = tf.convert_to_tensor(np.vstack(batch_paths_indices), dtype=tf.int32)
            ets_tensor = tf.convert_to_tensor(np.vstack(batch_ets_indices), dtype=tf.int32)
            tag_tensor = tf.convert_to_tensor(np.array(batch_tag_index, dtype=np.int64), dtype=tf.int64)
            yield (sts_tensor, paths_tensor, ets_tensor), tag_tensor


batch_losses = []
batch_accuracies = []


def on_batch_end(batch, logs):
    batch_losses.append(logs.get('loss'))
    batch_accuracies.append(logs.get('accuracy'))
    if batch % 200 == 0:
        print(f"Batch {batch}: Loss = {logs.get('loss'):.4f}, Accuracy = {logs.get('accuracy'):.4f}")

class TrainingTimeCallback(Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()  # Start the timer

    def on_train_end(self, logs=None):
        self.total_time = time.time() - self.start_time  # Calculate total time
        print(f"\n⏱️ Training completed in ({self.total_time/60:.4f} minutes)\n")

output_signature = (
    (tf.TensorSpec(shape=(None, max_num_contexts), dtype=tf.int32),  # for sts_indices
     tf.TensorSpec(shape=(None, max_num_contexts), dtype=tf.int32),  # for paths_indices
     tf.TensorSpec(shape=(None, max_num_contexts), dtype=tf.int32)),  # for ets_indices
    tf.TensorSpec(shape=(None,), dtype=tf.int64)  # for tag_index
)

dataset_train = tf.data.Dataset.from_generator(
    lambda: data_generator(train_file, batch_size),
    output_signature=output_signature
)

dataset_valid = tf.data.Dataset.from_generator(
    lambda: data_generator(valid_file, batch_size),
    output_signature=output_signature
)

training_timer = TrainingTimeCallback()
batch_logger = LambdaCallback(on_batch_end=on_batch_end)
steps_per_epoch = math.floor(number_lines_train / batch_size)  # Define the number of training batches per epoch
validation_steps = math.floor(number_lines_valid / batch_size)  

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=3,  
    restore_best_weights=True, 
    verbose=1
)
history = model.fit(
    dataset_train,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=dataset_valid,
    validation_steps=validation_steps,
    callbacks=[batch_logger, early_stopping, training_timer] 
)

model.save(f'trained_models/model_fold_{fold_idx}.h5')

print("\nTRAINING DONE!")

# -----------------------Plotting-------------------------------------------

plt.figure(figsize=(12, 6))

training_time_minutes = training_timer.total_time/60
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Loss per Epoch (Training Time: {training_time_minutes:.2f} min)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Accuracy per Epoch (Training Time: {training_time_minutes:.2f} min)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(f'analysis/learning_curves/learning_curve_fold_{fold_idx}.png')

plt.clf()


