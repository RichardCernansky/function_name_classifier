
import csv
import os
import sys

def filter_csv(input_file, column_name):
    # Create the output file name by appending "_only_c.csv" to the input file's base name
    base_name = os.path.splitext(input_file)[0]  # This removes the file extension
    output_file = f"{base_name}_only_c.csv"

    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)  # Use DictReader to access rows by column names
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        writer.writeheader()  # Write the header to the new file

        for row in reader:
            # Check if the value in the specified column ends with '.c'
            if row[column_name].strip().endswith('.c'):
                writer.writerow(row)  # Write the row to the output file if the condition is met

    print(f"Filtered rows saved to: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <input_file.csv> <column_name>")
        sys.exit(1)

    input_file = sys.argv[1]  # Get the input file from command-line argument

    filter_csv(input_file, "file") #second argument = column_name

