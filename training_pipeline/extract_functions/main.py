import os
import argparse
import csv
import subprocess
import sys
import json
import re
import hashlib

from pyparsing import line_end

from AsciiTreeProcessor import AsciiTreeProcessor
from NodeTree import NodeTree

# this is a script, that generates .ndjson file from dataset specified as parameter
# every line in .ndjson file represents tree structure of distinct string function
# (i.e. no function snippet occurs more than once in the .ndjson file (initial filtering))
# the functions' trees are obtained from ascii tree representation of every .c code snippet in the datasets

folder = None
file_name_col = None
code_snip_col = None

#consts
ndjson_path = "../data_ndjson/"
temp_file_path = "tmp/tempSourceCode.c"
ndjson_suffix = ".ndjson"

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

num_all_rows_c = 0
num_successful_rows = 0

seen_func_hashes = set()

def save_functions_to_ndjson(node_tree: NodeTree, ascii_tree, ndjson_path_t):
    """Save the entire tree as a single JSON object in NDJSON format."""
    global seen_func_hashes  # Access the global variable
    with open(ndjson_path_t, "a") as f:
        for child in node_tree.root_node.children:
            if child.kind == "FunctionDefinition":
                definition_node = child
                for definition_child in definition_node.children:
                    if definition_child.kind == "FunctionDeclarator":
                        declarator_node = definition_child

                        # Generate a hash for the tree
                        func_hash = AsciiTreeProcessor.hash_tree(declarator_node)

                        # Check if the function tree is new
                        if func_hash not in seen_func_hashes:
                            seen_func_hashes.add(func_hash)

                            # Process and save the function details
                            tag = declarator_node.data
                            declarator_node.data = "?"
                            func_tree_dict = definition_node.to_dict()
                            json_data = {
                                "tag": tag,
                                "num_tokens": AsciiTreeProcessor.get_num_tokens(definition_node),
                                "ast_depth": AsciiTreeProcessor.get_ast_depth(definition_node),
                                "num_nodes": AsciiTreeProcessor.get_num_nodes(definition_node),
                                "ast": func_tree_dict
                            }
                            json_line = json.dumps(json_data)
                            f.write(json_line + "\n")
                        break
                break

def ascii_to_ndjson(ascii_tree: str):
    """Convert an ASCII tree into NDJSON format."""
    atp = AsciiTreeProcessor(ascii_tree)
    node_tree = NodeTree(atp.produce_tree())
    global ndjson_path
    save_functions_to_ndjson(node_tree, ascii_tree, ndjson_path)

def run_cnip(prefix) -> subprocess.CompletedProcess[str]:
    """Run the CNIP command to generate the ASCII tree."""
    command = f"{prefix}psychec/cnip -l C -d {prefix}{temp_file_path}"
    return subprocess.run(command, shell=True, capture_output=True, text=True, encoding='ISO-8859-1')

def process_c_file(line: str):
    """Process a single C file."""
    global num_all_rows_c, num_successful_rows

    num_all_rows_c += 1

    with open(temp_file_path, 'w') as temp_file:
        temp_file.write(line)

    result = run_cnip("./")  # Run the external command

    # Check exit code and process the ASCII tree if successful
    if result.returncode == 0 and result.stdout.strip():
        num_successful_rows += 1
        ascii_to_ndjson(result.stdout)
    else:
        print("Error processing the function.")
#-------------------------------------------------------------------------------------------------------------------
#CSV process
def process_file_csv(csv_file_path: str):
    print(f"    Processing: {csv_file_path}.")

    # open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader((line.replace('\0', '') for line in file))

        for line in reader:
            # check if the specified filename_column ends with '.c', !!!Don't use .lower() for .C because some .C are cpp!!!
            if line[file_name_col] is not None and (line[file_name_col].endswith('.c') or line[file_name_col].lower().endswith('gnu c')):
                process_c_file(line[code_snip_col])

    success_rate = round(num_successful_rows / num_all_rows_c * 100, 2) if num_all_rows_c > 0 else 0
    print(f"        Finished processing: {csv_file_path}. Success rate: {success_rate}%. N.o. '.c' rows in csv: {num_all_rows_c}.")

def process_folder_csv(folder):
    print(f"Processing folder: {folder}.")
    # Loop through each .csv file in the folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                process_file_csv(str(csv_file_path))

#CSV process
#----------------------------------------------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Process files in a folder")

    # Define expected arguments
    parser.add_argument("folder", help="Folder path")
    parser.add_argument("file_name_col", help="File_name column name in csv")
    parser.add_argument("code_snip_col", help="Code_snippet column name in csv")


    return parser.parse_args()

def main():
    global folder, file_name_col, code_snip_col, ndjson_path  # Declare global variables

    args = get_args()

    folder = args.folder
    file_name_col = args.file_name_col
    code_snip_col = args.code_snip_col
    ndjson_path = ndjson_path + os.path.basename(folder) + ndjson_suffix
    print(ndjson_path)

    if os.path.exists(folder):
        process_folder_csv(folder)
    else:
        print(f"Error: Folder not found: {folder}")

#usage for gcj - main.py gcj file flines
#usage for codeforces - main.py codeforces language source_code
if __name__ == "__main__":
    main()
