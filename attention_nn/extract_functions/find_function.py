# python ./find_function.py /Users/richardcernansky/Desktop/bakalarka/datasets/gcj-dataset file flines --find_name dbsortfncsj
# python ./find_function.py /Users/richardcernansky/Desktop/bakalarka/datasets/gcj-dataset file flines --find_name llpow
# python ./find_function.py /Users/richardcernansky/Desktop/bakalarka/datasets/gcj-dataset file flines --find_name dbzt
# python ./find_function.py /Users/richardcernansky/Desktop/bakalarka/datasets/gcj-dataset file flines --find_name llcm


import os
import argparse
import csv
import re
import sys

folder = None
file_name_col = None
code_snip_col = None
find_name = None

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

# Function to extract function code by name from file
def find_function_by_name(file_path, target_name):
    function_pattern = re.compile(
        r'(unsigned|signed)?\s*(void|int|char|short|long|float|double)\s+\**(\w+)\s*\([^)]*\)\s*\{',
        re.MULTILINE
    )

    with open(file_path, 'r') as file:
        content = file.read()

    matches = function_pattern.finditer(content)

    for match in matches:
        name = match.group(3)
        if name == target_name:
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

    return None


def process_c_file(line: str):
    with open("tmp/tempSourceCode.c", 'w') as temp_file:
        temp_file.write(line)

    if find_name:
        function_code = find_function_by_name("tmp/tempSourceCode.c", find_name)
        if function_code:
            output_path = f"./tmp/{find_name}.txt"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Clear the file at the start of the program
            if not os.path.exists(output_path):
                open(output_path, 'w').close()
            with open(output_path, 'a') as output_file:
                output_file.write(function_code + "\n\n")
            print(f"Function {find_name} found and appended to {output_path}.")


# -------------------------------------------------------------------------------------------------------------------
# CSV process
def process_file_csv(csv_file_path: str):
    print(f"    Processing: {csv_file_path}.")

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader((line.replace('\0', '') for line in file))

        for line in reader:
            if line[file_name_col] is not None and (
                    line[file_name_col].endswith('.c') or line[file_name_col].lower().endswith('gnu c')):
                process_c_file(line[code_snip_col])


def process_folder_csv(folder):
    print(f"Processing folder: {folder}.")
    output_path = f"./tmp/{find_name}.txt"
    if find_name:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, 'w').close()  # Clear the file at the start of processing
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                process_file_csv(str(csv_file_path))


# CSV process
# ----------------------------------------------------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Process files in a folder")

    parser.add_argument("folder", help="Folder path")
    parser.add_argument("file_name_col", help="File_name column name in csv")
    parser.add_argument("code_snip_col", help="Code_snippet column name in csv")
    parser.add_argument("--find_name", help="Function name to find and print", default=None)

    return parser.parse_args()


def main():
    global folder, file_name_col, code_snip_col, find_name

    args = get_args()

    folder = args.folder
    file_name_col = args.file_name_col
    code_snip_col = args.code_snip_col
    find_name = args.find_name

    if os.path.exists(folder):
        process_folder_csv(folder)
    else:
        print(f"Error: Folder not found: {folder}")


if __name__ == "__main__":
    main()
