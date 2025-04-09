import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the path to the NDJSON file
ndjson_file = 'data_ndjson/train_valid_fold.ndjson'

# function to write both FunctionName, AST, and num_tokens columns to NDJSON format
def write_ndjson(file_path, df):
    with open(file_path, "w") as outfile:
        for _, row in df.iterrows():
            # Ensure the AST is properly formatted as JSON
            ast_json = row["AST"] if isinstance(row["AST"], dict) else json.loads(row["AST"])
            # Dump the fields to each line
            json_line = json.dumps({"tag": row["FunctionName"], "ast": ast_json, "num_tokens": row["NumTokens"], "source_code": row["SourceCode"]})
            outfile.write(json_line + "\n")


# load NDJSON data into `name_ast` list with error handling
name_ast = []
with open(ndjson_file, "r") as file:
    for line in file:
        try:
            function_json = json.loads(line.strip())
            function_name = function_json.get("tag")
            ast_node = function_json.get("ast")
            num_tokens = function_json.get("num_tokens")
            source_code = function_json.get("source_code")

            # Ensure function_name, ast, and num_tokens are valid before adding
            if function_name and ast_node and num_tokens is not None:
                name_ast.append({
                    "FunctionName": function_name,
                    "AST": json.dumps(ast_node),
                    "NumTokens": num_tokens,
                    "SourceCode": source_code
                })
        except json.JSONDecodeError:
            print(f"Error parsing line: {line}")

# create a DataFrame with the loaded data
df_name_ast = pd.DataFrame(name_ast)
print("Data loaded:")
print(df_name_ast.head())

# stratify the data into training and validation sets (75/25 split) based on the FunctionName
X = df_name_ast[['AST', 'NumTokens']]  # Features include AST and num_tokens
y = df_name_ast['FunctionName']  # Labels for stratification

train, valid = train_test_split(
    df_name_ast,
    test_size=0.1,
    stratify=y
)

# write the split data to NDJSON files with FunctionName, AST, and num_tokens fields
write_ndjson('data_ndjson/strat_train.ndjson', train)
write_ndjson('data_ndjson/strat_valid.ndjson', valid)

# verification
print(f"Training set size: {len(train)}")
print(f"Validation set size: {len(valid)}")
