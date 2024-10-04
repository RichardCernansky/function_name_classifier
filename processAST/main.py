import os
import argparse
import csv
import subprocess
import sys
import json
import re

from AsciiTreeProcessor import AsciiTreeProcessor
from NodeTree import NodeTree

#consts
ndjson_path = "ASTs.ndjson"
temp_file_path = "tmp/tempSourceCode.c"
# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

num_all_rows_c = 0
num_successful_rows = 0

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

def save_tree_to_ndjson(node_tree: NodeTree):
    """Save the entire tree as a single JSON object in NDJSON format."""
    with open(ndjson_path, "a") as f:
        # Convert the root node and its entire tree to a single dictionary
        tree_dict = node_tree.root_node.to_dict()

        #write to dictionary functions.ndjson if is function and is not main


        json_line = json.dumps(tree_dict)
        f.write(json_line + "\n")

def ascii_to_ndjson(ascii_tree: str):
    print(ascii_tree)
    atp = AsciiTreeProcessor(ascii_tree)
    node_tree = NodeTree(atp.produce_tree())
    save_tree_to_ndjson(node_tree)

def run_cnip() -> subprocess.CompletedProcess[str]:
    # Construct and execute the command
    command = f"./psychec/cnip -l C -d {temp_file_path}"
    return subprocess.run(command, shell=True, capture_output=True, text=True, encoding='ISO-8859-1')

def process_c_file(line: str):
    names_file_name = 'csv_names.txt'
    # names_file_name = 'json_names.txt'

    global num_all_rows_c, num_successful_rows
    num_all_rows_c += 1

    with open(temp_file_path, 'w') as temp_file:
        # Write the cleaned content to the temp file
        temp_file.write(line)

    result = run_cnip() #------------------------------------------RUN--------------------

    # check exitcode, if error -> thrash the tree (don't save it)
    if result.returncode != 0:
        pass
    # if successful, process the ascii-tree
    else:
        num_successful_rows += 1
        ascii_to_ndjson(result.stdout)
    return


#-------------------------------------------------------------------------------------------------------------------
#CSV process
def process_file_csv(csv_file_path: str, file_name_column: str):
    print(f"    Processing: {csv_file_path} with file_name_column: {file_name_column}")

    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader((line.replace('\0', '') for line in file))

        for line in reader:
            # Check if the specified filename_column ends with '.c'
            if line["file"].endswith('.c'):
                process_c_file(line["flines"])

    print(f"        Finished processing: {csv_file_path}. Success rate: {round(num_successful_rows/num_all_rows_c *100, 2)}%. N.o. rows in csv: {num_all_rows_c}.")

def process_folder_csv(folder, file_name_column):
    print(f"Processing folder: {folder} with file_name_column: {file_name_column}")
    # Loop through each .csv file in the folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                process_file_csv(str(csv_file_path), file_name_column)

#CSV process
#----------------------------------------------------------------------------------------------------------------------
#JSONL process

def process_file_jsonl(jsonl_file_path, file_name_column):
    print(f"Processing JSONL file: {jsonl_file_path}")

    # Open the .jsonl file
    with open(jsonl_file_path, mode='r', encoding='utf-8') as file:
        for line in file:
            try:
                # Parse the JSON line
                json_data = json.loads(line)

                # Extract the 'text' and 'func_def' fields
                if 'text' in json_data and 'func_def' in json_data['text']:
                    func_def_value = json_data['text'][file_name_column]
                    process_c_file(func_def_value)
                else:
                    print(f"'text' or 'func_def' key not found in the line.")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}")
                print(e)

    print(f"        Finished processing: {jsonl_file_path}. Success rate: {round(num_successful_rows / num_all_rows_c * 100, 2)}%. N.o. rows in csv: {num_all_rows_c}.")

def process_folder_jsonl(folder, file_name_column):
    print(f"Processing folder: {folder}")

    # Loop through each .jsonl file in the folder
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jsonl"):
                # Construct the full file path
                jsonl_file_path = os.path.join(root, file)
                # Process each jsonl file
                process_file_jsonl(jsonl_file_path, file_name_column)

#JSONL process
#--------------------------------------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Process files in a folder")

    # Define expected arguments
    parser.add_argument("folder", help="Folder path")
    parser.add_argument("file_name_column", help="File name column")
    parser.add_argument("file_type", help="Type of file: 'csv' or 'json'")

    return parser.parse_args()

def main():
    args = get_args()

    # Access arguments directly by their name
    folder = args.folder
    file_name_column = args.file_name_column
    file_type = args.file_type

    # Ensure the folder exists
    if os.path.exists(folder):
        if file_type == 'csv':
            process_folder_csv(folder, file_name_column)
        elif file_type == 'json':
            process_folder_jsonl(folder, file_name_column)
        else:
            print(f"Error: Unsupported file type '{file_type}'. Use 'csv' or 'json'.")
    else:
        print(f"Error: Folder not found: {folder}")

if __name__ == "__main__":
    main()
