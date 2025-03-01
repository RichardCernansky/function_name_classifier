import os
import sys
import ndjson
import re
import numpy as np
import pickle
import random
import time
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


from datasets import Dataset
import time
import evaluate
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath("/home/jovyan/function_name_classifier"))  # Add this modules path to sys


if len(sys.argv) < 2:
    print("Usage: python random_forest.py <fold_idx>")
    sys.exit(1)

fold_idx = int(sys.argv[1])

train_file = "data_ndjson/strat_train.ndjson"
valid_file = "data_ndjson/strat_valid.ndjson"
test_file = "data_ndjson/test_fold.ndjson"
vocabs_pkl = f'trained_models/vocabs_fold_{fold_idx}.pkl'


def preprocess_code(source_code):
    source_code = re.sub(r'//.*?$', '', source_code, flags=re.MULTILINE)  # Remove single-line comments
    source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)  # Remove multi-line comments
    source_code = re.sub(r'\s+', ' ', source_code).strip()  # Remove extra spaces
    return source_code

def load_ndjson(file_path):
    with open(file_path, "r") as f:
        data = ndjson.load(f)

    texts, labels = [], []
    for item in data:
        source_code = item["source_code"]
        author = item["tag"]
        texts.append(preprocess_code(source_code))
        labels.append(author)

    return texts, labels

train_texts, train_labels = load_ndjson(train_file)
test_texts, test_labels = load_ndjson(test_file)
print("Data loaded successfully.")

vectorizer = TfidfVectorizer(
    analyzer="word",  #code as a bag of words
    token_pattern=r"\b\w+\b",  # extract tokens
    max_features=5000  # limit features to 5000 most important tokens
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

label_encoder = LabelEncoder()

start_time = time.time()

y_train = label_encoder.fit_transform(train_labels)
y_test = label_encoder.transform(test_labels)
print("Data transformed successfully.")

param_grid = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

rf_model = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    rf_model, param_grid, n_iter=20, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
)

random_search.fit(X_train, y_train)

end_time = time.time()

print("Best Parameters:", random_search.best_params_)
# print("Best Accuracy:", random_search.best_score_)
print("Total training time:", (end_time - start_time)/60, " minutes")

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1: {f1:.4f}")

train_accuracy = best_model.score(X_train, y_train) 
test_accuracy = best_model.score(X_test, y_test) 
overfit_ratio = train_accuracy / test_accuracy
print(f"Overfit Ratio: {overfit_ratio:.2f}")


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="coolwarm", linewidths=0.5)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.savefig("conf_matrix")
