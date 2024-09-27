import os
import argparse
import csv
import subprocess
import sys
import json

from AsciiTreeProcessor import AsciiTreeProcessor
from NodeTree import NodeTree

#Consts
ndjson_path = "ASTs.ndjson"
temp_file_path = "tmp/tempSourceCode.c"
# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

def save_tree_to_ndjson(node_tree: NodeTree, filename: str):
    """Save the entire tree as a single JSON object in NDJSON format."""
    with open(filename, "a") as f:
        # Convert the root node and its entire tree to a single dictionary
        tree_dict = node_tree.root_node.to_dict()
        # Write the dictionary as a single JSON object (as one line in the NDJSON file)
        json_line = json.dumps(tree_dict)
        f.write(json_line + "\n")

def ascii_to_ndjson(ascii_tree: str):
    print(ascii_tree)
    atp = AsciiTreeProcessor(ascii_tree)
    node_tree = NodeTree(atp.produce_tree())
    save_tree_to_ndjson(node_tree, ndjson_path)

def run_cnip() -> subprocess.CompletedProcess[str]:
    # Construct and execute the command
    command = f"./psychec/cnip -l C -d {temp_file_path}"
    return subprocess.run(command, shell=True, capture_output=True, text=True, encoding='ISO-8859-1')

def process_c_file():
    pass

# Function to process each .csv file
def process_csv_file(csv_file_path: str, file_name_column: str):
    print(f"    Processing: {csv_file_path} with file_name_column: {file_name_column}")

    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader((line.replace('\0', '') for line in file))

        # Iterate over each line in the CSV
        num_all_rows_c = 0
        num_successful_rows = 0
        for line in reader:
            # Check if the specified filename_column ends with '.c'
            if line[file_name_column].endswith('.c'):
                num_all_rows_c += 1

                with open(temp_file_path, 'w') as temp_file:
                    # Write the cleaned content to the temp file
                    temp_file.write(line["flines"])

                result = run_cnip()

                #check exitcode, if error -> thrash the tree
                if result.returncode != 0:
                    pass
                #if successful, process the ascii-tree
                else:
                    num_successful_rows += 1
                    ascii_to_ndjson(result.stdout)

    print(f"        Finished processing: {csv_file_path}. Success rate: {round(num_successful_rows/num_all_rows_c*100, 2)}%. N.o. rows in csv: {num_all_rows_c}.")


# Main function to parse arguments and call processing
def main():
    parser = argparse.ArgumentParser(description="Process .csv files from dataset folders.")

    # Accept pairs of folders and file_name_column as arguments
    parser.add_argument("folder_column_pairs", nargs='+', help="Pairs of dataset folders and file_name_column.")

    args = parser.parse_args()

    # Ensure we have pairs (folder, file_name_column)
    if len(args.folder_column_pairs) % 2 != 0:
        print("Error: You must provide pairs of folder and file_name_column.")
        return

    # Iterate over the pairs
    for i in range(0, len(args.folder_column_pairs), 2):
        folder = args.folder_column_pairs[i]
        file_name_column = args.folder_column_pairs[i + 1]

        if os.path.exists(folder):
            print(f"Processing folder: {folder} with file_name_column: {file_name_column}")
            # Loop through each .csv file in the folder
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".csv"):
                        csv_file_path = os.path.join(root, file)
                        process_csv_file(str(csv_file_path), file_name_column)
        else:
            print(f"Folder not found: {folder}")


if __name__ == "__main__":
    main()
