import sys
import ndjson
import pickle
from NodeToNodePaths import json_to_tree, find_leaf_to_leaf_paths_iterative

if len(sys.argv) < 2:
    print("Usage: python generate_vocabs.py <fold_idx>")
    sys.exit(1)

fold_idx = int(sys.argv[1])  # Read and convert the fold index from command line argument

def generate_vocabs(file_paths):
    # Open the .ndjson file
    # Initialize empty sets
    value_vocab = set()  # Set of all leaf values
    path_vocab = set()  # Set of all distinct paths
    tags_vocab = set()  # Set of all distinct function tags
    max_num_contexts = 0
    for path in file_paths:
        with open(path, 'r') as ndjson_file:
            # Load the file content
            data = ndjson.load(ndjson_file)

            for function_json in data:
                tag = function_json['tag']
                func_tree_dict = function_json['ast']
                func_root = json_to_tree(func_tree_dict)

                func_values, func_paths = find_leaf_to_leaf_paths_iterative(func_root)
                max_num_contexts = max(len(func_paths), max_num_contexts)

                value_vocab.update(func_values)  # Add function's values to value_vocab set
                path_vocab.update(tuple(path[1:-1]) for path in func_paths)  # Add function's paths to path_vocab set
                tags_vocab.add(tag)

    value_vocab_dict = {value: idx + 1 for idx, value in enumerate(sorted(value_vocab))}
    path_vocab_dict = {path: idx + 1 for idx, path in enumerate(sorted(path_vocab))}
    tags_vocab_dict = {tag: idx + 1 for idx, tag in enumerate(sorted(tags_vocab))}

    # Append the padding values to the dictionaries
    value_vocab_dict['<PAD>'] = 0
    path_vocab_dict[('<PAD>',)] = 0
    tags_vocab_dict['<PAD>'] = 0

    # combine
    vocabs_dict = {
        'value_vocab': value_vocab_dict,
        'path_vocab': path_vocab_dict,
        'tags_vocab': tags_vocab_dict,
        'max_num_contexts': max_num_contexts
    }

    return vocabs_dict


vocabs_pkl = f'trained_models/vocabs_fold_{fold_idx}.pkl'
train = 'data_ndjson/strat_train.ndjson'
valid = 'data_ndjson/strat_valid.ndjson'

print("Started generating vocabs...")
vocabs_dict = generate_vocabs([train, valid])
with open(vocabs_pkl, 'wb') as f:
    pickle.dump(vocabs_dict, f)

print("Done.")