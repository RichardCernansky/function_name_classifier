from typing import Optional
import ndjson
import numpy as np
import pdb


import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import Input, Embedding, Dense, Activation, Lambda, Concatenate, Softmax, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import pad_sequences,plot_model
from tensorflow.keras.activations import softmax

from collections import OrderedDict #for ordered sets of the data

functionsASTs_file = 'functionsASTs.ndjson'



class Node:
    def __init__(self, b_i: Optional[int], kind: str, code_pos: str, data: str):
        self.branching_idx = b_i
        self.parent = None
        self.children = []
        self.kind = kind
        self.code_pos = code_pos
        self.data = data

    def set_parent(self, parent: 'Node'):
        self.parent = parent

    def add_child(self, child: 'Node'):
        self.children.append(child)

    def to_dict(self):
        """Convert the node and its children to a dictionary."""
        return {
            'kind': self.kind,
            'code_pos': self.code_pos,
            'data': self.data,
            'children': [child.to_dict() for child in self.children]
        }

def json_to_tree(data: dict) -> Node:
    """
    Recursively builds a tree of Node objects from a JSON dictionary.
    """
    node = Node(
        b_i=None,
        kind=data.get('kind'),
        code_pos=data.get('code_pos'),
        data=data.get('data')
    )

    # Recursively add children
    for child_data in data.get('children', []):
        child_node = json_to_tree(child_data)
        child_node.set_parent(node)  # Set the parent for the child node
        node.add_child(child_node)

    return node

#NODE TO NODE PATHS
# Function to collect all leaf nodes iteratively using DFS
def collect_leaves_iterative(root):
    if root is None:
        return []

    stack = [(root, [])]  # Stack to store (node, path_from_root)
    leaves = []  # List to store leaf nodes and their paths

    while stack:
        node, path = stack.pop()
        current_path = path + [node.kind]  # Update the current path

        # leaf node - has no children
        if not node.children:
            leaves.append((node, current_path))

        # push the children to the stack for DFS
        children = reversed(node.children)
        for child in children:  # process children in order on the stack
            stack.append((child, current_path))

    return leaves


# Function to find the Lowest Common Ancestor (LCA) iteratively
def find_lca_iterative(n1_path, n2_path):
    length = len(n1_path) if len(n1_path) < len(n2_path) else len(n2_path)

    lca = None
    for i in range(length):
        if n1_path[i] == n2_path[i]:
            lca = n1_path[i]
        else:
            break
    return lca


def find_leaf_to_leaf_paths_iterative(root):
    leaf_nodes = collect_leaves_iterative(root)

    #list of all leaf-to-leaf paths
    leaf_to_leaf_paths = []

    # Iterate over each pair of leaf nodes
    for i in range(len(leaf_nodes)):
        for j in range(i + 1, len(leaf_nodes)):
            leaf1, path1 = leaf_nodes[i]
            leaf2, path2 = leaf_nodes[j]

            # find lca
            lca = find_lca_iterative(path1, path2)

            # find the indexes
            lca_index1 = path1.index(lca)
            lca_index2 = path2.index(lca)

            # Path from leaf1 to leaf2 via the LCA
            path_to_lca_from_leaf1 = path1[:lca_index1 + 1]
            path_to_lca_from_leaf2 = path2[:lca_index2 + 1]
            path_to_lca_from_leaf2.reverse()

            #combine the paths
            complete_path = path_to_lca_from_leaf1 + path_to_lca_from_leaf2[1:]

            # Add the complete leaf-to-leaf path to the result
            leaf_to_leaf_paths.append((leaf1.data,)+tuple(complete_path)+(leaf2.data,))


    return [node.data for node,path in leaf_nodes], leaf_to_leaf_paths

def find_tag(root) -> str:
    # root is FunctionDefinition
    definition_node = root
    for definition_child in definition_node.children:
        if definition_child.kind == "FunctionDeclarator":
            declarator_node = definition_child
            for declarator_child in declarator_node.children:
                if declarator_child.kind == "IdentifierDeclarator":
                    return str(declarator_child.data)

    
def generate_vocabs(file_path):
    # Open the .ndjson file
    with open(file_path, 'r') as ndjson_file:
        # Load the file content
        data = ndjson.load(ndjson_file)

        # Initialize empty sets
        value_vocab = set()  # Set of all leaf values
        path_vocab = set()   # Set of all distinct paths
        tags_vocab = set()   # Set of all distinct function tags

        # Add '<PAD>' token before constructing the dictionaries
        value_vocab.add('<PAD>')
        path_vocab.add(('<PAD>',)) #tuple format
        tags_vocab.add('<PAD>')

        # Ensure that '<PAD>' gets index 0, and the other tokens start from index 1
        value_vocab_dict = {'<PAD>': 0}
        path_vocab_dict = {('<PAD>',): 0}
        tags_vocab_dict = {'<PAD>': 0}
        
        max_num_contexts = 0
        
        for function_json in data:
            # Convert each line (function) to a tree
            func_root = json_to_tree(function_json)
            tag = find_tag(func_root)
            func_values, func_paths = find_leaf_to_leaf_paths_iterative(func_root)
            max_num_contexts = max(len(func_paths), max_num_contexts)
            
            # Update vocabularies
            value_vocab.update(func_values)  # Add function's values to value_vocab set
            
            # Convert each list in func_paths to a tuple before updating the set
            path_vocab.update(path[1:-1] for path in func_paths)  # Add function's paths to path_vocab set
            
            tags_vocab.add(tag)  # add function's tag to tags_vocab set

        # create dictionaries from the sets by assigning each value an index
        value_vocab_dict = {value: idx+1 for idx, value in enumerate(sorted(value_vocab))}
        path_vocab_dict = {path: idx+1 for idx, path in enumerate(sorted(path_vocab))}
        tags_vocab_dict = {tag: idx+1 for idx, tag in enumerate(sorted(tags_vocab))}

        return value_vocab_dict, path_vocab_dict, tags_vocab_dict, max_num_contexts

            
            
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
        tags_embedding_matrix = self.tag_embedding(tf.range(self.tag_embedding.input_dim))  # Shape: (tags_vocab_size, embedding_dim)
        tags_embedding_matrix_t = tf.transpose(tags_embedding_matrix)  # Shape: (embedding_dim, tags_vocab_size)
        
        # num_repeats based on the shape of weighted_context
        num_repeats = tf.math.ceil( (tf.shape(inputs)[1] / tf.shape(tags_embedding_matrix_t)[0]))
        num_repeats = tf.cast(num_repeats, tf.int32)  # Ensure it's an integer

        # # tile it
        tags_embedding_matrix_t_tiled = tf.tile(tags_embedding_matrix_t, [num_repeats, 1])  # Shape: (num_repeats * embedding_dim, tags_vocab_size)
        
       
        
        # # only the required portion         # Shape: (weighted_context.shape[1], tags_vocab_size
        matrix_final  = tf.matmul( inputs, tags_embedding_matrix_t_tiled[:(tf.shape(inputs)[1])] )
        
        return matrix_final

def softmaxAxis1(x):
    return softmax(x,axis=1)


#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------START-TRAINING--------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
print("Started generating vocabs...")


value_vocab, path_vocab, tags_vocab, max_num_contexts = generate_vocabs(functionsASTs_file)

#vocab sizes and embedding dimensions
value_vocab_size = len(value_vocab)
path_vocab_size = len(path_vocab)
tags_vocab_size = len(tags_vocab)
embedding_dim = 128
y = embedding_dim #must be >= emb_dim

print("--------------------DONE--------------------")
print(f"value_vocab_size: {value_vocab_size}")
print(f"path_vocab_size: {path_vocab_size}")
print(f"tags_vocab_size: {tags_vocab_size}")
print(f"max_num_contexts: {max_num_contexts}")

print("Generating vocabs finished...")

# inputs for value1, path, and value2 (with num_context inputs per batch)
input_value1 = Input(shape=(max_num_contexts,), name='value1_input')
input_path = Input(shape=(max_num_contexts,), name='path_input')
input_value2 = Input(shape=(max_num_contexts,), name='value2_input')

# embedding layers with mask_zero=True to handle padding (index 0)
value_embedding = Embedding(input_dim=value_vocab_size, output_dim=embedding_dim, name='value_embedding', mask_zero=True)
path_embedding = Embedding(input_dim=path_vocab_size, output_dim=embedding_dim, name='path_embedding', mask_zero=True)

# embed the inputs
embedded_value1 = value_embedding(input_value1)  # shape: (None, num_context, embedding_dim)
embedded_path = path_embedding(input_path)      # shape: (None, num_context, embedding_dim)
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

#get tags_embeddings transposed
tag_scores = TagEmbeddingMatrixLayer(tags_vocab_size, embedding_dim)(weighted_context) 


output_tensor = Softmax()(tag_scores)

# define the model
model = Model(inputs=[input_value1, input_path, input_value2], outputs=output_tensor)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

#train
print("\n\nStarted training on functions...")

with open(functionsASTs_file, 'r') as ndjson_file:
    # Load the file content
    data = ndjson.load(ndjson_file)

    func_idx = 0
    for function_json in data:
        # Convert each line (function) to a tree
        func_root = json_to_tree(function_json)
        tag = find_tag(func_root)
        _, func_paths = find_leaf_to_leaf_paths_iterative(func_root) #get all contexts

        sts_indices = []     #start terminals indices
        paths_indices = []   #path indices
        ets_indices = []     #end terminals indices

        tag_index = tags_vocab[tag]  #get the tag value

        for path in func_paths: # map to the indices
            sts_indices.append(value_vocab[path[0]])    #get the terminal node's data
            paths_indices.append(path_vocab[path[1:-1]]) #get the path nodes' kinds
            ets_indices.append(value_vocab[path[-1]])   #get the ending terminal node's data


        # Use Keras `pad_sequences` for consistent padding to max_num_contexts
        sts_indices = pad_sequences([sts_indices], maxlen=max_num_contexts, padding='post', value=0)
        paths_indices = pad_sequences([paths_indices], maxlen=max_num_contexts, padding='post', value=0)
        ets_indices = pad_sequences([ets_indices], maxlen=max_num_contexts, padding='post', value=0)


        # Convert inputs to the right data type (int64) if needed
        sts_indices = np.array(sts_indices, dtype=np.int64)
        paths_indices = np.array(paths_indices, dtype=np.int64)
        ets_indices = np.array(ets_indices, dtype=np.int64)
        tag_index = np.array([tag_index], dtype=np.int64)  # Ensure tag_index has batch dimension


        # Perform a training step and capture only the loss
        loss_values = model.train_on_batch(x=[sts_indices, paths_indices, ets_indices], y=tag_index)

        # Assuming the first value in the returned list is the loss
        loss = loss_values[0]

        # Print the loss value
        print(f"\tFunction {func_idx} successfully trained! Loss: {loss:.4f}")

        func_idx += 1



print("Training done!")



