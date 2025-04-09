import os
import argparse
import csv
import subprocess
import sys
import json
import re
from dataclasses import dataclass
from typing import Set

from AsciiTreeProcessor import AsciiTreeProcessor
from NodeTree import NodeTree

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)


@dataclass
class ProcessorConfig:
    folder: str
    file_name_col: str
    code_snip_col: str
    author_col: str
    ndjson_path: str
    temp_file_path: str = "tmp/tempSourceCode.c"


class CSVProcessor:
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.num_all_rows_c = 0
        self.num_successful_rows = 0
        self.seen_func_strings: Set[str] = set()

    def save_functions_to_ndjson(self, source_code: str, node_tree: NodeTree, author_name: str):
        """Save the entire tree as a single JSON object in NDJSON format."""
        ndjson_path_t = self.config.ndjson_path
        with open(ndjson_path_t, "a") as f:
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
                                        "ast": func_tree_dict,
                                        "source_code": source_code
                                    }
                                    f.write(json.dumps(json_data) + "\n")
                                    break
                            break

    def ascii_to_ndjson(self, source_code:str,  ascii_tree: str, author_name: str):
        """Convert an ASCII tree into NDJSON format."""
        atp = AsciiTreeProcessor(ascii_tree)
        node_tree = NodeTree(atp.produce_tree())
        self.save_functions_to_ndjson(source_code, node_tree, author_name)

    def run_cnip(self, prefix) -> subprocess.CompletedProcess:
        """Run the CNIP command to generate the ASCII tree."""
        command = f"{prefix}psychec/cnip -l C -d {prefix}{self.config.temp_file_path}"
        return subprocess.run(command, shell=True, capture_output=True, text=True, encoding='ISO-8859-1')

    def extract_func_body(self, content, match):
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

    def process_c_file(self, line_func: str, author_name: str):
        function_pattern = re.compile(
            r'^\s*(unsigned|signed)?\s*(void|int|char|short|long|float|double)\s+\**(\w+)\s*\([^)]*\)\s*\{',
            re.MULTILINE
        )

        matches = function_pattern.finditer(line_func)
        for match in matches:
            self.num_all_rows_c += 1
            func_string = self.extract_func_body(line_func, match)
            if func_string not in self.seen_func_strings:
                self.seen_func_strings.add(func_string)
                with open(self.config.temp_file_path, 'w') as temp_file:
                    temp_file.write(func_string)

                result = self.run_cnip("./")
                if result.returncode == 0 and result.stdout.strip():
                    self.num_successful_rows += 1
                    
                    self.ascii_to_ndjson(func_string, result.stdout, author_name)
                else:
                    print(f"Error processing function:\n{func_string}")

    def process_file_csv(self, csv_file_path: str):
        """Processes a single CSV file."""
        print(f"    Processing: {csv_file_path}.")

        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader((line.replace('\0', '') for line in file))

            for line in reader:
                if line[self.config.file_name_col] and (
                        line[self.config.file_name_col].endswith('.c') or
                        line[self.config.file_name_col].lower().endswith('gnu c')):
                    self.process_c_file(line[self.config.code_snip_col], line[self.config.author_col])

        success_rate = round(self.num_successful_rows / self.num_all_rows_c * 100, 2) if self.num_all_rows_c > 0 else 0
        print(f"Finished processing: {csv_file_path}. Success rate: {success_rate}%. Processed rows: {self.num_all_rows_c}.")

    def process_folder_csv(self):
        """Processes all CSV files in a folder."""
        print(f"Processing folder: {self.config.folder}.")
        for root, _, files in os.walk(self.config.folder):
            for file in files:
                if file.endswith(".csv"):
                    csv_file_path = os.path.join(root, file)
                    self.process_file_csv(csv_file_path)


def get_args():
    parser = argparse.ArgumentParser(description="Process files in a folder")
    parser.add_argument("folder", help="Folder path")
    parser.add_argument("file_name_col", help="File_name column name in CSV")
    parser.add_argument("code_snip_col", help="Code_snippet column name in CSV")
    parser.add_argument("author_col", type=str, help="Author's name")
    return parser.parse_args()


def main():
    args = get_args()

    config = ProcessorConfig(
        folder=args.folder,
        file_name_col=args.file_name_col,
        code_snip_col=args.code_snip_col,
        author_col=args.author_col,
        ndjson_path=f"../data_ndjson/{os.path.basename(args.folder)}.ndjson"
    )

    processor = CSVProcessor(config)

    if os.path.exists(config.folder):
        processor.process_folder_csv()
    else:
        print(f"Error: Folder not found: {config.folder}")


if __name__ == "__main__":
    main()
