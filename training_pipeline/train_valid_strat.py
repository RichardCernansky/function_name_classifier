import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the path to the NDJSON file
ndjson_file = 'data_ndjson/train_valid_fold.ndjson'

# Function to write both FunctionName and AST columns to NDJSON format
def write_ndjson(file_path, df):
    with open(file_path, "w") as outfile:
        for _, row in df.iterrows():
            # Ensure the AST is properly formatted as JSON
            ast_json = row["AST"] if isinstance(row["AST"], dict) else json.loads(row["AST"])
            # Dump both FunctionName and AST fields to each line
            json_line = json.dumps({"tag": row["FunctionName"], "ast": ast_json})
            outfile.write(json_line + "\n")

# Load NDJSON data into `name_ast` list with error handling
name_ast = []
with open(ndjson_file, "r") as file:
    for line in file:
        try:
            function_json = json.loads(line.strip())
            function_name = function_json.get("tag")
            ast_node = function_json.get("ast")
            if function_name and ast_node:
                # Convert AST to a JSON string for consistent handling
                name_ast.append({"FunctionName": function_name, "AST": json.dumps(ast_node)})
        except json.JSONDecodeError:
            print(f"Error parsing line: {line}")

# Create a DataFrame with the loaded data
df_name_ast = pd.DataFrame(name_ast)
print("Data loaded:")
print(df_name_ast.head())

# Stratify the data into training and validation sets (75/25 split)
train, valid = train_test_split(df_name_ast, test_size=0.25, stratify=df_name_ast["FunctionName"])

# Write the split data to NDJSON files with both FunctionName and AST fields
write_ndjson('data_ndjson/strat_train.ndjson', train)
write_ndjson('data_ndjson/strat_valid.ndjson', valid)

# Verification
print(f"Training set size: {len(train)}")
print(f"Validation set size: {len(valid)}")

