import os
import argparse
import csv
import subprocess
import sys
from typing import List

#Consts
temp_file_path = "tmp/tempSourceCode.c"
# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

class Node:
    def __init__(self, b_i: int):
        self.branching_idx = b_i
        self.parent = None
        self.children = []
        self.kind = None
        self.code_pos = None
        self.data = None

    def set_parent(self, parent: 'Node'):
        self.parent = parent

    def add_child(self, child: 'Node'):
        self.children.append(child)


class AsciiTreeProcessor:
    def __init__(self, tree: str):
        self.lines = self.remove_empty_back(tree.split("\n")[1:])

    def visit(self, line_idx: int, cur_node: Node):
        if line_idx >= len(self.lines):
            return
        else:
            b_i = cur_node.branching_idx
            line_b_i = self.lines[line_idx].find('|--')
            if line_b_i > b_i:
                new_node = Node(line_b_i)
                new_node.set_parent(cur_node)
                cur_node.add_child(new_node)
                self.visit(line_idx + 1, new_node)
                return
            elif line_b_i == b_i:
                new_node = Node(line_b_i)
                cur_node.parent.add_child(new_node)
                new_node.set_parent(cur_node.parent)
                self.visit(line_idx + 1, new_node)
                return
            else:
                self.visit(line_idx, cur_node.parent)
                return

    def remove_empty_back(self, lines: List[str]) -> List[str]:
        idx = 0
        while lines[len(lines) - 1 - idx] == "":
            lines.pop()
        return lines



def ascii_to_json(ascii_tree: str):
    print(ascii_tree)
    atp = AsciiTreeProcessor(ascii_tree)
    root_node = Node(-1)
    atp.visit(1, root_node)

    return

def run_cnip():
    # Construct and execute the command
    command = f"./psychec/cnip -l C -d {temp_file_path}"  # >/dev/null 2>/dev/null"
    return subprocess.run(command, shell=True, capture_output=True, text=True, encoding='ISO-8859-1')

def process_c_file():
    pass

# Function to process each .csv file
def process_csv_file(csv_file_path: str, file_name_column: str):
    print(f"    Processing: {csv_file_path} with file_name_column: {file_name_column}")

    # Open the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        reader = csv.DictReader((line.replace('\0', '') for line in file))

        # Iterate over each line in the CSV
        num_all_rows_c = 0
        num_successful_rows = 0
        for line in reader:
            # Check if the specified filename_column ends with '.c'
            if line[file_name_column].lower().endswith('.c'):
                num_all_rows_c += 1

                with open(temp_file_path, 'w') as temp_file:
                    # Write the cleaned content to the temp file
                    temp_file.write(line["flines"])

                result = run_cnip()

                #check exitcode, if error -> thrash the tree
                if result.returncode != 0:
                    pass
                else:
                    num_successful_rows += 1
                    ascii_to_json(result.stdout)

    print(f"        Finished processing: {csv_file_path}. Success rate: {round(num_successful_rows/num_all_rows_c*100, 2)}%")


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
                        process_csv_file(csv_file_path, file_name_column)
        else:
            print(f"Folder not found: {folder}")


if __name__ == "__main__":
    main()
