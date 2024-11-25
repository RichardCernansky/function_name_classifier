import json
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import os
from scipy.stats import shapiro, kstest

def get_basename_without_extension(path_string):
    return os.path.splitext(os.path.basename(path_string))[0]

#input_ndjson_file = "../data_ndjson/functionsASTs_gcj-dataset.ndjson"
#input_ndjson_file = "../data_ndjson/functionsASTs_contests.ndjson"
input_ndjson_file = "../data_ndjson/functionsASTs_merged.ndjson"
output_ndjson_file = "../data_ndjson/functionsASTs_dropped_lower_5.ndjson"
# output_csv_file = get_basename_without_extension(input_ndjson_file) + "_freq_table.csv"
output_freq_histogram_pdf_file = get_basename_without_extension(input_ndjson_file) + "_freq_histogram.pdf"
output_length_histogram_pdf_file = get_basename_without_extension(input_ndjson_file) + "_length_histogram.pdf"

poor_names = ['main', 'solve']

function_names = []
function_lines = []  # Store lines for future filtering
function_lengths_tokens = []
with open(input_ndjson_file, "r") as file:
    for line in file:
        try:
            function_json = json.loads(line.strip())
            function_name = function_json.get('tag')
            root_ast_node = function_json.get('ast')
            num_tokens = function_json.get('num_tokens')
            if function_name and root_ast_node and num_tokens:
                function_names.append(function_name)
                function_lines.append((function_name, line, num_tokens))
        except json.JSONDecodeError:
            print(f"Error parsing line: {line}")

function_counter = Counter(function_names)

filtered_function_names = set()
data = []

for function, freq in function_counter.items():
    if freq >= 5 and function not in poor_names: #filter
        filtered_function_names.add(function)
        data.append({"FunctionName": function, "Frequency": freq})

total_functions = sum(item["Frequency"] for item in data)

for item in data:
    item["Percentage"] = round((item["Frequency"] / total_functions) * 100, 2)

df = pd.DataFrame(data)
df = df.sort_values(by="Frequency", ascending=False)

df.columns = [f"FunctionName", f"Frequency (Total: {total_functions})", "Percentage"]

# WRITE NEW .NDJSON
filtered_function_lengths_tokens = []
with open(output_ndjson_file, "w") as outfile:
    for function_name, original_line,num_tokens in function_lines:
        if function_name in filtered_function_names:
            outfile.write(original_line)
            filtered_function_lengths_tokens.append(num_tokens)

#prepare df
df = pd.DataFrame(data)
print(df.head())
# sort the dataframe by frequency in descending order
df = df.sort_values(by="Frequency", ascending=False)

#Shapiro-Wilk test on frequencies
shapiro_stat, shapiro_p = shapiro(df["Frequency"])
print(f"Shapiro-Wilk Test Results:")
print(f"  W Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4e}")
if shapiro_p < 0.05:
    print("The data is not normally distributed (p < 0.05).")
else:
    print("The data follows a normal distribution (p >= 0.05).")

column_name = f"Frequency (Total: {total_functions})"
df.columns = ["FunctionName", column_name, "Percentage"]

#PLOT NAMES
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
plt.savefig(output_freq_histogram_pdf_file, format='pdf')
plt.tight_layout()  # Adjust layout to fit labels nicely
plt.show()

print(len(function_lengths_tokens))
ks_statistic, ks_p_value = kstest(function_lengths_tokens, 'norm')
# Print the results
print(f"Kolmogorov-Smirnov Test Results:")
print(f"  KS Statistic: {ks_statistic:.4f}")
print(f"  P-value: {ks_p_value:.4e}")

if ks_p_value < 0.05:
    print("The data is not normally distributed (p < 0.05).")
else:
    print("The data follows a normal distribution (p >= 0.05).")

plt.figure(figsize=(12, 6), dpi=100)
plt.hist(function_lengths_tokens, bins=100, range=(0, 500), edgecolor='black')  # Reduce bin count for better clarity
plt.xlim(0, 500)  # Focus on functions with lengths up to 500 tokens
plt.grid(axis='y', alpha=0.75)
plt.title('Focused Histogram of Function Lengths in Tokens (0-500)')
plt.xlabel('Function Length (number of tokens)')
plt.ylabel('Number of Functions')
plt.xticks(range(0, 500 + 50, 50), fontsize=10, rotation=45)
plt.tight_layout()  # Ensure labels are not cut off
plt.savefig("FOCUSED_" + output_length_histogram_pdf_file, format='pdf')
plt.show()