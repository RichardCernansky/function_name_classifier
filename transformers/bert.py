import os
import sys
import ndjson
import numpy as np
import pickle
import random
import time
from collections import Counter
import matplotlib.pyplot as plt

from datasets import Dataset
import evaluate
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

sys.path.append(os.path.abspath("/home/jovyan/function_name_classifier"))  # Add this modules path to sys

from training_pipeline.extract_functions import Node
from training_pipeline.NodeToNodePaths import json_to_tree, pre_order_traversal


if len(sys.argv) < 2:
    print("Usage: python bert.py <fold_idx>")
    sys.exit(1)

fold_idx = int(sys.argv[1])

MAX_LENGTH_IN_TOKENS = 512
train_file = "data_ndjson/strat_train.ndjson"
valid_file = "data_ndjson/strat_valid.ndjson"
test_file = "data_ndjson/test_fold.ndjson"
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
        for func_json in data:
            tag = func_json.get("tag")
            ast_json = func_json.get("ast")
            
            
            func_root = json_to_tree(ast_json)
            traversal = pre_order_traversal(func_root)  # Get all contexts

            tokens_joined = " ".join(traversal)
            author_id = tags_vocab[tag]

            data_dict = {"ast_paths": tokens_joined, "author": author_id}
            bert_data.append(data_dict)

        return bert_data


_, _, tags_vocab, _ = get_vocabs(vocabs_pkl)
inverted_tags_vocab = {value: key for key,value in tags_vocab.items()}
train_data = get_data(tags_vocab, train_file)
valid_data = get_data(tags_vocab, valid_file)
test_data = get_data(tags_vocab, test_file)

train_labels = [d["author"] for d in train_data]
train_label_counts = Counter(train_labels)
sorted_train_labels = sorted(train_label_counts.items(), key=lambda x: x[1], reverse=True)

print("Sorted Train Label Distribution:", sorted_train_labels)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")


def tokenize_function(example):
    tokenized_inputs = tokenizer(example["ast_paths"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = example["author"]  # Ensure labels are included
    return tokenized_inputs


# Convert to Hugging Face Dataset
dataset1 = Dataset.from_list(train_data)
dataset2 = Dataset.from_list(valid_data)
dataset3 = Dataset.from_list(test_data)

# Tokenize dataset
train_dataset = dataset1.map(tokenize_function, batched=True)
val_dataset = dataset2.map(tokenize_function, batched=True)
test_dataset = dataset3.map(tokenize_function, batched=True)

num_authors = len(tags_vocab)
print("Number of distinct authors: ", num_authors-1)

model = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=num_authors)

# Freeze parameters
for param in model.roberta.parameters():
    param.requires_grad = False

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    # you can just use the first two.
    if isinstance(eval_pred, (tuple, list)):
        logits, labels = eval_pred[0], eval_pred[1]
    else:
        # Otherwise, assume it's an EvalPrediction-like object
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

start_time = time.time()

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=300,
    save_total_limit=2,
    metric_for_best_model="accuracy", 
    greater_is_better=True  
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=50  # Stop training if val accuracy does not improve for 5 epochs
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]  
)

# Start training
trainer.train()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n⏱️ Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)\n")

train_preds = trainer.predict(train_dataset) 
train_metrics = compute_metrics(train_preds)
train_acc = train_metrics["accuracy"]
val_metrics = trainer.evaluate()  
val_acc = val_metrics["eval_accuracy"]

overfit_ratio = train_acc / val_acc
print("Overfit ratio: ", overfit_ratio)

log_history = trainer.state.log_history
train_losses = [entry['loss'] for entry in log_history if 'loss' in entry]
val_losses   = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
val_acc      = [entry['eval_accuracy'] for entry in log_history if 'eval_accuracy' in entry]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.savefig("learning_curve.png")

model.save_pretrained("./bert-authorship")
tokenizer.save_pretrained("./bert-authorship")












