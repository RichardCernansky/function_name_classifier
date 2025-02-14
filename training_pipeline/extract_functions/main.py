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
author_col = None

# consts
ndjson_path = "../data_ndjson/"
temp_file_path = "tmp/tempSourceCode.c"
ndjson_suffix = ".ndjson"

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

num_all_rows_c = 0
num_successful_rows = 0

seen_func_strings = set()


def save_functions_to_ndjson(node_tree: NodeTree, ndjson_path_t: str, author_name: str):
    # print(ascii_tree) #ascii tree for debug prints
    """Save the entire tree as a single JSON object in NDJSON format."""
    with open(ndjson_path_t, "a") as f:
        # write to dictionary functions.ndjson if is function and is not ['main', 'solve']
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
                                    "tag": tag if author_name == "" else author_name,
                                    "num_tokens": AsciiTreeProcessor.get_num_tokens(definition_node),
                                    "ast_depth": AsciiTreeProcessor.get_ast_depth(definition_node),
                                    "num_nodes": AsciiTreeProcessor.get_num_nodes(definition_node),
                                    "ast": func_tree_dict
                                }
                                json_line = json.dumps(json_data)
                                f.write(json_line + "\n")
                                break
                        break


def ascii_to_ndjson(ascii_tree: str, author_name: str):
    """Convert an ASCII tree into NDJSON format."""
    atp = AsciiTreeProcessor(ascii_tree)
    node_tree = NodeTree(atp.produce_tree())
    global ndjson_path
    save_functions_to_ndjson(node_tree, ndjson_path, author_name)


def run_cnip(prefix) -> subprocess.CompletedProcess[str]:
    """Run the CNIP command to generate the ASCII tree."""
    command = f"{prefix}psychec/cnip -l C -d {prefix}{temp_file_path}"
    return subprocess.run(command, shell=True, capture_output=True, text=True, encoding='ISO-8859-1')


def extract_func_body(content, match):
    start = match.start()
    bracket_count = 1
    end = start + len(match.group())
    while end < len(content) and bracket_count > 0:
        if content[end] == '{':
            bracket_count += 1
        elif content[end] == '}':
            bracket_count -= 1
        end += 1
    return content[start:end]


def process_c_file(line_func: str, author_name: str):
    global num_all_rows_c, num_successful_rows, seen_func_strings

    function_pattern = re.compile(
        r'^\s*(unsigned|signed)?\s*(void|int|char|short|long|float|double)\s+\**(\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE
    )

    matches = function_pattern.finditer(line_func)
    for match in matches:
        num_all_rows_c += 1

        func_string = extract_func_body(line_func, match)
        if func_string not in seen_func_strings:
            seen_func_strings.add(func_string)
            with open(temp_file_path, 'w') as temp_file:
                temp_file.write(func_string)

            result = run_cnip("./")

            if result.returncode == 0 and result.stdout.strip():
                num_successful_rows += 1
                ascii_to_ndjson(result.stdout, author_name)
            else:
                print(f"Error processing function:\n{func_string}")


# -------------------------------------------------------------------------------------------------------------------
# CSV process
def process_file_csv(csv_file_path: str):
    print(f"    Processing: {csv_file_path}.")

    # open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader((line.replace('\0', '') for line in file))

        for line in reader:
            # check if the specified filename_column ends with '.c', !!!Don't use .lower() for .C because some .C are cpp!!!
            if line[file_name_col] is not None and (
                    line[file_name_col].endswith('.c') or line[file_name_col].lower().endswith('gnu c')):
                process_c_file(line[code_snip_col], line[author_col])

    success_rate = round(num_successful_rows / num_all_rows_c * 100, 2) if num_all_rows_c > 0 else 0
    print(
        f"        Finished processing: {csv_file_path}. Success rate: {success_rate}%. N.o. '.c' rows in csv: {num_all_rows_c}.")


def process_folder_csv(folder):
    print(f"Processing folder: {folder}.")
    # Loop through each .csv file in the folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                process_file_csv(str(csv_file_path))


# CSV process
# ----------------------------------------------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Process files in a folder")

    # Define expected arguments
    parser.add_argument("folder", help="Folder path")
    parser.add_argument("file_name_col", help="File_name column name in csv")
    parser.add_argument("code_snip_col", help="Code_snippet column name in csv")
    parser.add_argument("author_col", type=str, help="Author's name (requires --author)")

    return parser.parse_args()


def main():
    global folder, file_name_col, code_snip_col, ndjson_path, author_col  # Declare global variables

    args = get_args()

    folder = args.folder
    file_name_col = args.file_name_col
    code_snip_col = args.code_snip_col
    author_col = args.author_col

    ndjson_path = ndjson_path + os.path.basename(folder) + ndjson_suffix
    print(ndjson_path)

    if os.path.exists(folder):
        process_folder_csv(folder)
    else:
        print(f"Error: Folder not found: {folder}")


# usage for gcj - main.py gcj file flines
# usage for codeforces - main.py codeforces language source_code
if __name__ == "__main__":
    main()