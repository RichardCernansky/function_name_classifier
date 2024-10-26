import json
import math
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

from NodeToNodePaths import find_tag

ndjson_file = 'data_ndjson/train_valid_fold.ndjson'

# Function to write only the AST column to NDJSON format
def write_ndjson(file_path, df):
    with open(file_path, "w") as outfile:
        for _, row in df.iterrows():
            # Write only the AST field
            ast_json = json.loads(row["AST"])  # Convert the string back to JSON if needed
            json_line = json.dumps(ast_json)   # Dump only the AST
            outfile.write(json_line + "\n")    # Write each AST as a separate line from the df

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

# Create a DataFrame with the loaded data
df_name_ast = pd.DataFrame(name_ast)
print(df_name_ast.head())

# Stratify the data into training, validation, and test sets
train, valid = train_test_split(df_name_ast, test_size=0.25, stratify=df_name_ast["FunctionName"])

# Write the split data to NDJSON files with only the AST field
write_ndjson('data_ndjson/strat_train.ndjson', train)
write_ndjson('data_ndjson/strat_valid.ndjson', valid)

# Verification
print(f"Training set size: {len(train)}")
print(f"Validation set size: {len(valid)}")
