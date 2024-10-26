import json
import pandas as pd
from collections import Counter
from NodeToNodePaths import find_tag
from matplotlib import pyplot as plt

input_ndjson_file = "data_ndjson/functionsASTs.ndjson"
output_ndjson_file = "data_ndjson/functionsASTs_dropped_lower_5.ndjson"

function_names = []
function_lines = []  # Store lines for future filtering
with open(input_ndjson_file, "r") as file:
    for line in file:
        try:
            root_ast_node = json.loads(line.strip())
            function_name = find_tag(root_ast_node)
            if function_name:
                function_names.append(function_name)
                function_lines.append((function_name, line))  # keep track of original lines
        except json.JSONDecodeError:
            print(f"Error parsing line: {line}")

function_counter = Counter(function_names)

filtered_function_names = set()
data = []

for function, freq in function_counter.items():
    if freq >= 5: #filter
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


df = pd.DataFrame(data)
print(df.head())
# Sort the dataframe by frequency in descending order
df = df.sort_values(by="Frequency", ascending=False)
column_name = f"Frequency (Total: {total_functions})"
df.columns = ["FunctionName", column_name, "Percentage"]
df.to_csv("analysis_csv/freq_analysis_gcj_dropped.csv", index=False)

# Plotting the bar graph
plt.figure(figsize=(20, 8))  # Adjust figure size as needed

# Bar plot to display each function name and its frequency
plt.bar(df["FunctionName"], df[column_name], color='blue', edgecolor='black')

# Adding titles and labels
plt.title('Frequencies of function names')
plt.xlabel('Function Name')
plt.ylabel('Frequency')

plt.yticks(range(0, int(df[column_name].max() + 50), 50))
plt.xticks(rotation=45, ha='right', fontsize=1)

# Add grid lines to improve readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the plot
plt.savefig("function_name_frequencies_histogram.pdf", format='pdf')
plt.tight_layout()  # Adjust layout to fit labels nicely
plt.show()

