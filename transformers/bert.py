import os
import sys
import ndjson
import numpy as np
import pickle
import random

from datasets import Dataset
import evaluate
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

sys.path.append(os.path.abspath("/home/jovyan/function_name_classifier")) #add this modules path to sys

from training_pipeline.extract_functions import Node
from training_pipeline.NodeToNodePaths import json_to_tree, find_leaf_to_leaf_paths_iterative

if len(sys.argv) < 2:
    print("Usage: python bert.py <fold_idx>")
    sys.exit(1)

fold_idx = int(sys.argv[1])  

NUM_SAMPLES_PER_FUNCTION = 200
MAX_LENGTH_IN_TOKENS = 512
train_file = "data_ndjson/strat_train.ndjson"
valid_file = "data_ndjson/strat_valid.ndjson"
vocabs_pkl = f'trained_models/vocabs_fold_{fold_idx}.pkl'


def get_vocabs(vocabs_pkl):
    with open(vocabs_pkl, 'rb') as f:
        vocabs = pickle.load(f)
        return vocabs['value_vocab'], vocabs['path_vocab'], vocabs['tags_vocab'], vocabs['max_num_contexts']

def get_data(tags_vocab: dict, data_file):
    "outputs data in the form of joined function's paths with space and id of author"
    with open(data_file, 'r') as ndjson_file:
        data = ndjson.load(ndjson_file)
        np.random.shuffle(data)

        bert_data = []
        tokens_total = 0
        for func_json in data:
            tag = func_json.get("tag")
            ast_json = func_json.get("ast")
            func_root = json_to_tree(ast_json)
            _, func_paths = find_leaf_to_leaf_paths_iterative(func_root)  # get all contexts
            
            num_samples = min(NUM_SAMPLES_PER_FUNCTION, len(func_paths))
            sampled_func_paths = random.sample(func_paths, num_samples)

            paths_tokens = []
            for node_node_path in sampled_func_paths:
                paths_tokens.append(list(node_node_path[1:-1]))
                
            flatten_paths_tokens = [item for sublist in paths_tokens for item in sublist]
            tokens_total += len(flatten_paths_tokens)

            tokens = " ".join(flatten_paths_tokens)
            author_id = tags_vocab[tag]

            data_dict = {"ast_paths": tokens, "author": author_id}
            bert_data.append(data_dict)
            
        print("average num of tokens per function: ", tokens_total/len(data))
        return bert_data   
        
_, _, tags_vocab, _ = get_vocabs(vocabs_pkl)      
train_data = get_data(tags_vocab, train_file)
valid_data = get_data(tags_vocab, valid_file)


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")


def tokenize_function(example):
    tokenized_inputs = tokenizer(example["ast_paths"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = example["author"]  # Ensure labels are included
    return tokenized_inputs

# Convert to Hugging Face Dataset
dataset1 = Dataset.from_list(train_data)
dataset2 = Dataset.from_list(valid_data)

# Tokenize dataset
train_dataset = dataset1.map(tokenize_function, batched=True)
val_dataset = dataset2.map(tokenize_function, batched=True)

num_authors = len(tags_vocab)

model = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=num_authors)
for param in model.roberta.parameters():
    param.requires_grad = False

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)  # Get predicted class index
    return metric.compute(predictions=predictions, references=labels)
    
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    save_total_limit=2
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

model.save_pretrained("./codebert-authorship")
tokenizer.save_pretrained("./codebert-authorship")