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
				./scripts/clearErrorASTs.sh "$csv_file" 2>/dev/null 
    fi
done

