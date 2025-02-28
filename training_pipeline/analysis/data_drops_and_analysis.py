import json
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
import os
from scipy.stats import shapiro,kstest, norm 

def get_basename_without_extension(path_string):
    return os.path.splitext(os.path.basename(path_string))[0]

names_prefix = "exploratory_analysis/names/"
token_lengths_prefix = "exploratory_analysis/token_lengths/"
ast_depths_prefix = "exploratory_analysis/ast_depths/"
num_nodes_prefix = "exploratory_analysis/num_nodes/"

pdf_postfix = ".pdf"
input_ndjson_file = "../data_ndjson/gcj-dataset.ndjson"
#input_ndjson_file = "../data_ndjson/contests.ndjson"
#input_ndjson_file = "../data_ndjson/merged_without_noise.ndjson"
output_ndjson_file = "../data_ndjson/dropped_lower_10.ndjson"
basename_without_extension = get_basename_without_extension(input_ndjson_file)

output_names_histogram_pdf_file = names_prefix + basename_without_extension + pdf_postfix
output_lengths_histogram_pdf_file = token_lengths_prefix + basename_without_extension + pdf_postfix
output_depths_pdf_file = ast_depths_prefix + basename_without_extension + pdf_postfix
output_num_nodes_pdf_file = num_nodes_prefix + basename_without_extension + pdf_postfix

MINIMAL_FREQUENCY = 40
# poor_names = ['main', 'solve']
poor_names = ['Curse','stubbscroll', 'dahlukeh','baihacker','eduardische', 'trainsickYao', 'p5ic05i5','WhiteShadow', 'daybreakcx','dwagndwagn']

#FETCH DATA
function_names = []
function_lines = []  # Store lines for future filtering
with open(input_ndjson_file, "r") as file:
    for line in file:
        try:
            function_json = json.loads(line.strip())
            function_name = function_json.get('tag')
            root_ast_node = function_json.get('ast')
            num_tokens = function_json.get('num_tokens')
            ast_depth = function_json.get('ast_depth')
            num_nodes = function_json.get('num_nodes')
            if function_name and root_ast_node and num_tokens and ast_depth and num_nodes:
                function_names.append(function_name)
                function_lines.append((function_name, line, num_tokens, ast_depth, num_nodes))
        except json.JSONDecodeError:
            print(f"Error parsing line: {line}")

#FILTER BASED ON THE COUNTER
function_counter = Counter(function_names)
filtered_function_names = set()
data = []

for function, freq in function_counter.items():
    if (
        freq >= MINIMAL_FREQUENCY and
        not any(poor_name.lower() in function.lower() for poor_name in poor_names)
    ):
        filtered_function_names.add(function)
        data.append({"FunctionName": function, "Frequency": freq})


total_functions = sum(item["Frequency"] for item in data)

for item in data:
    item["Percentage"] = round((item["Frequency"] / total_functions) * 100, 2)

df = pd.DataFrame(data)
df = df.sort_values(by="Frequency", ascending=False)
df.columns = [f"FunctionName", f"Frequency (Total: {total_functions})", "Percentage"]

# WRITE NEW IN FILTERED .NDJSON
filtered_lengths_tokens = []
filtered_ast_depths = []
filtered_num_nodes = []
with open(output_ndjson_file, "w") as outfile:
    for function_name, original_line,num_tokens, ast_depth,num_nodes in function_lines:
        if function_name in filtered_function_names:
            outfile.write(original_line)
            filtered_lengths_tokens.append(num_tokens)
            filtered_ast_depths.append(ast_depth)
            filtered_num_nodes.append(num_nodes)

#prepare df
df = pd.DataFrame(data)
print(df.head())
# sort the dataframe by frequency in descending order
df = df.sort_values(by="Frequency", ascending=False)

#NAMES
#Shapiro-Wilk test on name frequencies
mean = np.mean(df["Frequency"])
std = np.std(df["Frequency"])
shapiro_stat, shapiro_p = shapiro(df["Frequency"])
print(f"NAMES: Shapiro-Wilk Test Results:")
print(f"  W Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4e}")
if shapiro_p < 0.05:
    print("The data is not normally distributed (p < 0.05).")
else:
    print("The data follows a normal distribution (p >= 0.05).")
print(f"mean: {mean:.4f}, std: {std:.4f}")

import matplotlib.pyplot as plt

# Ensure df is defined (replace this with your actual DataFrame)
# df = pd.DataFrame(...)

# Define column names dynamically
column_name = f"Frequency (Total: {total_functions})"
df.columns = ["FunctionName", column_name, "Percentage"]
plt.figure(figsize=(20, 8))  
bars = plt.bar(df["FunctionName"], df[column_name], color='blue', edgecolor='black')
for bar, value in zip(bars, df[column_name]):
    plt.text(bar.get_x() + bar.get_width() / 2,  # Center text
             value + 2,  # Slightly above bar
             f"{int(value)}",  # Format as integer
             ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")
plt.title(f'Distribution of Number of Functions per Author. Total number of functions = {total_functions}', fontsize=16, fontweight="bold")

plt.xlabel('Function Name', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.yticks(range(0, int(df[column_name].max() + 50), 50))
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_names_histogram_pdf_file, format='pdf')



#NUM_TOKENS
#Kolmogorov-Smirnov test for num_tokens
mean = np.mean(filtered_lengths_tokens)
std = np.std(filtered_lengths_tokens)
ks_statistic, ks_p_value = kstest(filtered_lengths_tokens, 'norm', args=(mean, std))
print(f"NUM_TOKENS: Kolmogorov-Smirnov Test Results:")
print(f"  KS Statistic: {ks_statistic:.4f}")
print(f"  P-value: {ks_p_value:.4e}")
if ks_p_value < 0.05:
    print("The data is not normally distributed (p < 0.05).")
else:
    print("The data follows a normal distribution (p >= 0.05).")
print(f"mean: {mean:.4f}, std: {std:.4f}")

plt.figure(figsize=(12, 6), dpi=100)
counts, bins, patches = plt.hist(filtered_lengths_tokens, bins=100, range=(0, 500), edgecolor='black')
for count, bin_edge in zip(counts, bins[:-1]):  # Exclude the last bin edge
    if count > 0:  # Avoid labeling empty bins
        plt.text(bin_edge + (bins[1] - bins[0]) / 2,  # Center text above bar
                 count + 1,  # Slightly above the bar
                 f"{int(count)}",  # Format as integer
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")
plt.xlim(0, 500)  # Focus on functions with token_lengths up to 500 tokens
plt.grid(axis='y', alpha=0.75)
plt.title('Focused Histogram of Function Lengths in Tokens (0-500)', fontsize=14, fontweight="bold")
plt.xlabel('Function Length (number of tokens)', fontsize=12)
plt.ylabel('Number of Functions', fontsize=12)
plt.xticks(range(0, 500 + 50, 50), fontsize=10, rotation=45)
plt.tight_layout() 
plt.savefig(output_lengths_histogram_pdf_file, format='pdf')


#AST_DEPTH
mean = np.mean(filtered_ast_depths)
std = np.std(filtered_ast_depths)
ks_statistic, ks_p_value = kstest(filtered_ast_depths, 'norm', args=(mean, std))
print(f"AST_DEPTHS: Kolmogorov-Smirnov Test Results:")
print(f"  KS Statistic: {ks_statistic:.4f}")
print(f"  P-value: {ks_p_value:.4e}")
if ks_p_value < 0.05:
    print("The data is not normally distributed (p < 0.05).")
else:
    print("The data follows a normal distribution (p >= 0.05).")
print(f"mean: {mean:.4f}, std: {std:.4f}")

max_depth = int(max(filtered_ast_depths))
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(filtered_ast_depths, bins=max_depth, edgecolor="black", alpha=0.7)

# Add frequency values above bars
for count, bin_edge in zip(counts, bins[:-1]):  # Exclude the last bin edge
    if count > 0:  # Avoid labeling empty bins
        plt.text(bin_edge + (bins[1] - bins[0]) / 2,  # Center text above bar
                 count + 1,  # Slightly above the bar
                 f"{int(count)}",  # Format as integer
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")
plt.title("Distribution of Filtered AST Depths", fontsize=14, fontweight="bold")
plt.xlabel("AST Depth", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(range(0, max_depth + 2, 2))
plt.tight_layout()
plt.savefig(output_depths_pdf_file, format='pdf')


#NUM_NODES
mean = np.mean(filtered_num_nodes)
std = np.std(filtered_num_nodes)
ks_statistic, ks_p_value = kstest(filtered_num_nodes, 'norm', args=(mean, std))
print(f"NUM_NODES: Kolmogorov-Smirnov Test Results:")
print(f"  KS Statistic: {ks_statistic:.4f}")
print(f"  P-value: {ks_p_value:.4e}")
if ks_p_value < 0.05:
    print("The data is not normally distributed (p < 0.05).")
else:
    print("The data follows a normal distribution (p >= 0.05).")
print(f"mean: {mean:.4f}, std: {std:.4f}")

upper_limit = max(filtered_num_nodes)
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(filtered_num_nodes, bins=70, edgecolor="black", alpha=0.7)
# add labels above bars
for count, bin_edge in zip(counts, bins[:-1]):  # Exclude the last bin edge
    if count > 0:  # Avoid labeling empty bins
        plt.text(bin_edge + (bins[1] - bins[0]) / 2,  # Center text above bar
                 count + 1,  # Slightly above the bar
                 f"{int(count)}",  # Format as integer
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="black")
plt.title("Distribution of Filtered Number of Nodes", fontsize=14, fontweight="bold")
plt.xlabel("Number of Nodes", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, upper_limit + 50, 50), fontsize=8, rotation=45)
plt.xlim(0, upper_limit)
plt.tight_layout() 
plt.savefig(output_num_nodes_pdf_file, format='pdf')


