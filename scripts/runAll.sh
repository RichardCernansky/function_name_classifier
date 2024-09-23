#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Get the directory from the argument
dataset_dir="$1"

# Check if the dataset directory exists
if [ ! -d "$dataset_dir" ]; then
    echo "Dataset directory does not exist: $dataset_dir"
    exit 1
fi

# Loop through all CSV files in the datasets directory
for csv_file in "$dataset_dir"/*_only_c.csv; do
    if [ -f "$csv_file" ]; then
        # Run the script on each CSV file, suppressing all output (stdout and stderr)
				(./psycheC_ast/build/psycheC_ast ".csv"  "$csv_file" ) 2>/dev/null
    fi

		# Get the total number of lines
		total_lines=$(wc -l < ./row_index.log)
		# Get the last line (assumed to be a number)
		last_line=$(tail -n1 ./row_index.log)
		# Calculate the percentage (total_lines / last_line) * 100
		percentage=$(awk "BEGIN {printf \"%.2f\", 100 - ($total_lines / $last_line) * 100}")
		# Output the result
		echo "File: $csv_file. Percentage of successfully created ASTs: $percentage%"

done

