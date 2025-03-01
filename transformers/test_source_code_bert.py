import os
import sys
import ndjson
import numpy as np
import pickle
import random
import time
import re
from collections import Counter
import matplotlib.pyplot as plt
import torch
import seaborn as sns

from datasets import Dataset
import evaluate
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath("/home/jovyan/function_name_classifier"))  # Add this modules path to sys

from training_pipeline.extract_functions import Node
from training_pipeline.NodeToNodePaths import json_to_tree, pre_order_traversal

def get_vocabs(vocabs_pkl):
    with open(vocabs_pkl, 'rb') as f:
        vocabs = pickle.load(f)
        return vocabs['value_vocab'], vocabs['path_vocab'], vocabs['tags_vocab'], vocabs['max_num_contexts']

def tokenize_function(example):
    tokenized_inputs = tokenizer(example["source_code"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = example["author"]  # Ensure labels are included
    return tokenized_inputs

def preprocess_code(source_code):
    source_code = re.sub(r'//.*?$', '', source_code, flags=re.MULTILINE)  # Remove single-line comments
    source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)  # Remove multi-line comments
    source_code = re.sub(r'\s+', ' ', source_code).strip()  # Remove extra spaces
    return source_code

def get_data(tags_vocab: dict, data_file):
    "outputs data in the form of joined function's paths with space and id of author"
    with open(data_file, 'r') as ndjson_file:
        data = ndjson.load(ndjson_file)
        np.random.shuffle(data)

        bert_data = []
        for func_json in data:
            tag = func_json.get("tag")
        
            tokens_joined = preprocess_code(func_json.get("source_code"))
            author_id = tags_vocab[tag]

            data_dict = {"source_code": tokens_joined, "author": author_id}
            bert_data.append(data_dict)

        return bert_data

#-----TEST------
vocabs_pkl = f'trained_models/vocabs_fold_1.pkl'
_, _, tags_vocab, _ = get_vocabs(vocabs_pkl)
inverted_tags_vocab = {value: key for key,value in tags_vocab.items()}
model_path = "./codebert-authorship"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model.eval()

test_file = "data_ndjson/test_fold.ndjson"
test_data = get_data(tags_vocab, test_file)
dataset3 = Dataset.from_list(test_data)
test_dataset = dataset3.map(tokenize_function, batched=True)

def predict(model, tokenized_data):
    input_ids = torch.tensor(tokenized_data["input_ids"])
    attention_mask = torch.tensor(tokenized_data["attention_mask"])
    
    with torch.no_grad():  
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits 
    predictions = torch.argmax(logits, dim=-1)  # Get highest probability class index
    return predictions.cpu().numpy()

predicted_labels = predict(model, test_dataset)
decoded_predictions = [inverted_tags_vocab[idx] for idx in predicted_labels]
true_labels = [example["author"] for example in test_data]

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1: {f1:.4f}")




conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="coolwarm", linewidths=0.5)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.savefig("conf_matrix")