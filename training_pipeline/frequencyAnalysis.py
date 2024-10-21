import json
import math
import pandas as pd
from collections import Counter

from NodeToNodePaths import find_tag

ndjson_file = "data_ndjson/functionsASTs_dropped_singles_doubles.ndjson"
train_perc = 0.7
valid_perc = 0.3

def write_ndjson(file_name, array):
    with open(file_name, 'w') as f:
        for item in array:
            f.write(item)

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

train = []
validation = []
test = []

for function_name, freq in function_counter.items():
    train_size = math.floor(freq * train_perc)
    valid_size = math.floor(freq * valid_perc)
    # test_size = all - train - valid

    train_seen = 0
    valid_seen = 0
    for item in name_ast:
        item_name = item['FunctionName']
        if item_name == function_name:
            if train_seen < train_size:
                train.append(item['AST'])
                train_seen += 1
            elif valid_seen < valid_size:
                validation.append(item['AST'])
                valid_seen += 1
            else:
                test.append(item['AST'])

# WRITING FILES
write_ndjson('data_ndjson/strat_train_functionsASTs.ndjson', train)
write_ndjson('data_ndjson/strat_validate_functionsASTs.ndjson', validation)
write_ndjson('data_ndjson/strat_test_functionsASTs.ndjson', test)

# verification
print(f"Training set size: {len(train)}")
print(f"Validation set size: {len(validation)}")
print(f"Test set size: {len(test)}")

df = pd.DataFrame(name_freq)
df = df.sort_values(by="Frequency", ascending=False)
df.columns = [f"FunctionName", f"Frequency (Total: {total_functions})", "Percentage"]
df.to_csv("analysis_csv/freq_analysis_gcj_dropped.csv", index=False)
df.head()
