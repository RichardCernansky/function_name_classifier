import pandas as pd
import json
import os
from sklearn.model_selection import StratifiedKFold
import subprocess

# Sample input file (replace with actual file path)
ndjson_file = "data_ndjson/functionsASTs_dropped_lower_5.ndjson"
#clear the .log file contents before the whole process
with open("analysis_csv/tests_results.log", "a") as log_file:
    log_file.write("")

# Load the data from NDJSON
name_ast = []
with open(ndjson_file, "r") as file:
    for line in file:
        try:
            function_json = json.loads(line.strip())
            function_name = function_json.get("tag")
            ast = function_json.get("ast")

            # Ensure both function_name and ast are valid before adding
            if function_name and ast:
                name_ast.append({"FunctionName": function_name, "AST": json.dumps(ast)})
            else:
                print(f"Missing 'tag' or 'ast' in line (skipped): {line}")

        except json.JSONDecodeError:
            print(f"Error parsing line (skipped): {line}")

# Create DataFrame and verify contents
df_name_ast = pd.DataFrame(name_ast)
if "FunctionName" not in df_name_ast.columns or df_name_ast.empty:
    print("Error: DataFrame is empty or missing 'FunctionName' column.")
else:
    print("Data loaded successfully:")
    print(df_name_ast.head())

    # Proceed with Stratified K-Fold if DataFrame is correctly populated
    X = df_name_ast['AST'].values  # Features (AST as strings)
    y = df_name_ast['FunctionName'].values  # Labels for stratification

    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

    for fold_index, (train_valid_index, test_index) in enumerate(strat_kfold.split(X, y)):
        X_train_valid, X_test = X[train_valid_index], X[test_index]
        y_train_valid, y_test = y[train_valid_index], y[test_index]

        train_valid_ndjson_file = f'data_ndjson/train_valid_fold.ndjson'
        test_ndjson_file = f'data_ndjson/test_fold.ndjson'

        # Save train/validation set
        with open(train_valid_ndjson_file, 'w') as outfile:
            for ast_str, func_name in zip(X_train_valid, y_train_valid):
                json.dump({"tag": func_name, "ast": json.loads(ast_str)}, outfile)
                outfile.write('\n')

        # Save test set
        with open(test_ndjson_file, 'w') as outfile:
            for ast_str, func_name in zip(X_test, y_test):
                json.dump({"tag": func_name, "ast": json.loads(ast_str)}, outfile)
                outfile.write('\n')

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

