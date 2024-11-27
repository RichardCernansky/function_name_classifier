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

# this is a program, that generates .ndjson file from dataset specified as parameter
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


# get hash for seen_func_set
def get_md5_hash(input_string: str) -> str:
    md5_object = hashlib.md5()
    md5_object.update(input_string.encode('utf-8'))
    return md5_object.hexdigest()

#function to extract function from string stored in file_path
def extract_function_names(file_path):
    # might be broken by some complicated  function pointer arguments, or macros and so on...
    function_pattern = re.compile(
        r'^\s*(unsigned|signed)?\s*(void|int|char|short|long|float|double)\s+\**(\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE
    )

    with open(file_path, 'r') as file:
        content = file.read()

    # Find all function names using the pattern
    function_names = function_pattern.findall(content)

    return function_names

def save_functions_to_ndjson(node_tree: NodeTree, ascii_tree, ndjson_path_t):
    #print(ascii_tree) #ascii tree for debug prints
    """Save the entire tree as a single JSON object in NDJSON format."""
    with open(ndjson_path_t, "a") as f:
        #write to dictionary functions.ndjson if is function and is not ['main', 'solve']
        for child in node_tree.root_node.children:
            if child.kind == "FunctionDefinition":
                definition_node = child
                for definition_child in definition_node.children:
                    if definition_child.kind == "FunctionDeclarator":
                        declarator_node = definition_child
                        for declarator_child in declarator_node.children:
                            if declarator_child.kind == "IdentifierDeclarator":
                                tag = declarator_child.data
                                declarator_child.data = "?"
                                func_tree_dict = definition_node.to_dict()
                                json_data = {
                                    "tag": tag,
                                    "num_tokens": AsciiTreeProcessor.get_num_tokens(definition_node),
                                    "ast_depth": AsciiTreeProcessor.get_ast_depth(definition_node),
                                    "num_leaves": AsciiTreeProcessor.get_num_leaves(definition_node),
                                    "ast": func_tree_dict
                                }
                                json_line = json.dumps(json_data)
                                f.write(json_line + "\n")
                                break
                        break

def ascii_to_ndjson(ascii_tree: str):
    # print(ascii_tree)
    atp = AsciiTreeProcessor(ascii_tree)
    node_tree = NodeTree(atp.produce_tree())
    global ndjson_path
    save_functions_to_ndjson(node_tree, ascii_tree, ndjson_path)

def run_cnip(prefix) -> subprocess.CompletedProcess[str]:
    # Construct and execute the command
    command = f"{prefix}psychec/cnip -l C -d {prefix}{temp_file_path}"
    return subprocess.run(command, shell=True, capture_output=True, text=True, encoding='ISO-8859-1')

def process_c_file(line: str, seen_func_hashes: set):

    global num_all_rows_c, num_successful_rows
    num_all_rows_c += 1

    func_hash = get_md5_hash(line)
    if line not in seen_func_hashes:
        seen_func_hashes.add(func_hash)

        with open(temp_file_path, 'w') as temp_file:
            # Write the cleaned content to the temp file
            temp_file.write(line)

        result = run_cnip("./") #------------------------------------------RUN--------------------

        # check exitcode, if error -> thrash the tree (don't save it)
        if result.returncode != 0:
            pass
        # if successful, process the ascii-tree
        else:
            num_successful_rows += 1
            ascii_to_ndjson(result.stdout)
        return
    else:
        print("Repeated function found.")
        pass #don't do anything with function we already saw


#-------------------------------------------------------------------------------------------------------------------
#CSV process
def process_file_csv(csv_file_path: str, seen_func_hashes: set):
    print(f"    Processing: {csv_file_path}.")

    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader((line.replace('\0', '') for line in file))

        for line in reader:
            # Check if the specified filename_column ends with '.c', !!!Don't use .lower() for .C because some .C are cpp!!!
            if line[file_name_col] is not None and (line[file_name_col].endswith('.c') or line[file_name_col].lower().endswith('gnu c')):
                process_c_file(line[code_snip_col], seen_func_hashes)

    success_rate = round(num_successful_rows/num_all_rows_c *100, 2) if num_all_rows_c > 0 else 0
    print(f"        Finished processing: {csv_file_path}. Success rate: {success_rate}%. N.o. '.c' rows in csv: {num_all_rows_c}.")


def process_folder_csv(folder, seen_func_hashes: set):
    print(f"Processing folder: {folder}.")
    # Loop through each .csv file in the folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                process_file_csv(str(csv_file_path), seen_func_hashes)

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
    global folder, file_name_col, code_snip_col, ndjson_path  # Declare that we are using the global variables

    args = get_args()

    folder = args.folder
    file_name_col = args.file_name_col
    code_snip_col = args.code_snip_col
    ndjson_path = ndjson_path + os.path.basename(folder) + ndjson_suffix
    print(ndjson_path)

    seen_func_hashes = set()
    if os.path.exists(folder):
        process_folder_csv(folder, seen_func_hashes)
    else:
        print(f"Error: Folder not found: {folder}")

#usage for gcj - main.py gcj file flines
#usage for codeforces - main.py codeforces language source_code
if __name__ == "__main__":
    main()
