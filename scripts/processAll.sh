#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Get the directory from the argument
target_dir="$1"

# Check if the specified directory exists
if [ ! -d "$target_dir" ]; then
    echo "Directory does not exist: $target_dir"
    exit 1
fi

# Loop through all .tar.bz2 files in the target directory
for tar_file in "$target_dir"/*.tar.bz2; do
    if [ -f "$tar_file" ]; then
        # Extract the base name of the file (removes the extension)
        base_name=$(basename "$tar_file" .tar.bz2)
        csv_file="$target_dir/$base_name"

        # Extract the .tar.bz2 file
        echo "Extracting $tar_file..."
        tar -xjf "$tar_file" -C "$target_dir"

        # Check if the CSV file exists after extraction
        if [ -f "$csv_file" ]; then
            echo "Processing CSV file: $csv_file"

            # Run the filter_for_csv.py script for the CSV file
            python3 filter_csv_for_c.py "$csv_file"

            echo "Finished processing $csv_file."
        else
            echo "Error: Expected CSV file $csv_file not found after extracting $tar_file"
        fi
    fi
done

