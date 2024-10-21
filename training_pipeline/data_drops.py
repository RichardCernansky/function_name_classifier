import json
import pandas as pd
from collections import Counter
from NodeToNodePaths import find_tag

input_ndjson_file = "data_ndjson/functionsASTs.ndjson"
output_ndjson_file = "data_ndjson/functionsASTs_dropped_singles_doubles.ndjson"

function_names = []
function_lines = []  # Store lines for future filtering

with open(input_ndjson_file, "r") as file:
    for line in file:
        try:
            ast_node = json.loads(line.strip())
            function_name = find_tag(ast_node)
            if function_name:
                function_names.append(function_name)
                function_lines.append((function_name, line))  # Keep track of original lines
        except json.JSONDecodeError:
            print(f"Error parsing line: {line}")

function_counter = Counter(function_names)

filtered_function_names = set()
data = []

for function, freq in function_counter.items():
    if freq >= 3: #filter
        filtered_function_names.add(function)
        data.append({"FunctionName": function, "Frequency": freq})

total_functions = sum(item["Frequency"] for item in data)

for item in data:
    item["Percentage"] = round((item["Frequency"] / total_functions) * 100, 2)

df = pd.DataFrame(data)
df = df.sort_values(by="Frequency", ascending=False)

df.columns = [f"FunctionName", f"Frequency (Total: {total_functions})", "Percentage"]

# WRITE NEW .NDJSON
with open(output_ndjson_file, "w") as outfile:
    for function_name, original_line in function_lines:
        if function_name in filtered_function_names:
            outfile.write(original_line)


# Display the DataFrame
print(df.head())

