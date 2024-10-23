import json
import math
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

from NodeToNodePaths import find_tag

ndjson_file = "data_ndjson/functionsASTs_dropped_singles_doubles.ndjson"

# Function to write DataFrame to NDJSON format
def write_ndjson(file_path, df):
    with open(file_path, "w") as outfile:
        for _, row in df.iterrows():
            json_line = json.dumps({"FunctionName": row["FunctionName"], "AST": row["AST"]})
            outfile.write(json_line + "\n")

function_names = []
name_ast = []
with open(ndjson_file, "r") as file:
    for line in file:
        try:
            ast_node = json.loads(line.strip())
            function_name = find_tag(ast_node)
            if function_name:
                function_names.append(function_name)
                name_ast.append({"FunctionName": function_name, "AST": line})
        except json.JSONDecodeError:
            print(f"Error parsing line: {line}")

function_counter = Counter(function_names)
total_functions = sum(function_counter.values())

name_freq = []
for function_name, freq in function_counter.items():
    percentage = (freq / total_functions) * 100
    name_freq.append({"FunctionName": function_name, "Frequency": freq, "Percentage": round(percentage, 2)})

df_name_ast = pd.DataFrame(name_ast)
print(df_name_ast.head())

# Stratify the data into training, validation, and test sets
train, temp = train_test_split(df_name_ast, test_size=0.4, stratify=df_name_ast["FunctionName"])
validation, test = train_test_split(temp, test_size=0.4, stratify=temp["FunctionName"])

# Write the split data to NDJSON files
write_ndjson('data_ndjson/strat_train_functionsASTs.ndjson', train)
write_ndjson('data_ndjson/strat_validate_functionsASTs.ndjson', validation)
write_ndjson('data_ndjson/strat_test_functionsASTs.ndjson', test)

# Verification
print(f"Training set size: {len(train)}")
print(f"Validation set size: {len(validation)}")
print(f"Test set size: {len(test)}")


