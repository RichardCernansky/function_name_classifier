#!/bin/bash
     
# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
		echo "Usage: $0 <input_file.csv \"datasets/gcj2008.csv\">"
    exit 1
fi


# Get the input file from the first argument
input_file="$1"
relative_path="$input_file"
last_index=0
program_log="psycheC_ast.log"
row_index_log="row_index.log"

# Clear or create the row_index.log file at the start
> "$row_index_log"

while true; do
		#DEBUG PRINTS
    #if (( last_index % 17 == 0 )); then
        #echo "Starting from row_index: $last_index"
    #fi
    #echo "Starting from row_index: $last_index"

    # Run the executable and capture the output in a log file, suppressing stderr
    ./psycheC_ast/build/psycheC_ast "$relative_path" "$last_index" > "$program_log" 2>/dev/null
    exit_code=$?  # Capture the exit status of the program


    # Check if the program reached the end of the file by looking for "EOF" in the log
    if grep -q "EOF" "$program_log"; then
        #echo "End of file reached. Exiting."
        break
    fi

    # Check if the program exited due to a segmentation fault (exit code 139)
    if [ $exit_code -eq 139 ]; then
	last_index=$(grep "row_index" "$program_log" | tail -n1 | awk '{print $2}')
        echo "$last_index" >> "$row_index_log"
        last_index=$((last_index + 1))
    fi
done


# Get the total number of lines
total_lines=$(wc -l < "$row_index_log")
# Get the last line (assumed to be a number)
last_line=$(tail -n1 "$row_index_log")
# Calculate the percentage (total_lines / last_line) * 100
percentage=$(awk "BEGIN {printf \"%.2f\", 100 - ($total_lines / $last_line) * 100}")
# Output the result
echo "File: $input_file. Percentage of successfully created ASTs: $percentage%"
