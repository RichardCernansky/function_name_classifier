import pandas as pd
import json
import os
from sklearn.model_selection import StratifiedKFold
from NodeToNodePaths import find_tag
import subprocess

# Sample input file (replace with actual file path)
ndjson_file = "data_ndjson/functionsASTs_dropped_lower_5.ndjson"
#clear the .log file contents before the whole process
with open("analysis_csv/tests_results.log", "w") as log_file:
    log_file.write("")

# Load the data from NDJSON
name_ast = []
with open(ndjson_file, "r") as file:
    for line in file:
        try:
            ast_node = json.loads(line.strip())
            function_name = find_tag(ast_node)
            if function_name:
                name_ast.append({"FunctionName": function_name, "AST": line})
        except json.JSONDecodeError:
            print(f"Error parsing line: {line}")

# Create a DataFrame
df_name_ast = pd.DataFrame(name_ast)
print("Data loaded:")
print(df_name_ast.head())

# Define your features and labels for stratification
X = df_name_ast['AST'].values  # Features (the actual AST data as strings)
y = df_name_ast['FunctionName'].values  # Labels (the function names for stratification)

strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

fold_index = 0
for train_valid_index, test_index in strat_kfold.split(X, y):
    # split the data into training/validation and test sets
    X_train_valid, X_test = X[train_valid_index], X[test_index]

    # Define file paths for the current fold
    train_valid_ndjson_file = 'data_ndjson/train_valid_fold.ndjson'
    test_ndjson_file = 'data_ndjson/test_fold.ndjson'

    # Save the ASTs from the training/validation set
    with open(train_valid_ndjson_file, 'w') as outfile:
        for ast_str in X_train_valid:
            ast_json = json.loads(ast_str)  # Convert the string back to a JSON object
            json.dump(ast_json, outfile)
            outfile.write('\n')

    # Save the ASTs from the test set
    with open(test_ndjson_file, 'w') as outfile:
        for ast_str in X_test:
            ast_json = json.loads(ast_str)  # Convert the string back to a JSON object
            json.dump(ast_json, outfile)
            outfile.write('\n')

    # Print the current fold being processed
    print(f"Fold {fold_index} saved as NDJSON files.")

    # Run external scripts using the generated files
    try:
        subprocess.run(["python", "train_valid_strat.py"], check=True)
        # Generate vocabs for the current fold
        subprocess.run(["python", "generate_vocabs.py"], check=True)
        # Train the model on the current fold
        subprocess.run(["python", "AttentionCNNClassifier.py", str(fold_index+1)], check=True)
        # Test the model on the current fold's test file
        subprocess.run(["python", "testing_model.py", str(fold_index+1)], check=True)

        print(f"Completed processing for Fold {fold_index}")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred during processing of Fold {fold_index}: {e}")

    fold_index += 1

print("All folds processed successfully!")
