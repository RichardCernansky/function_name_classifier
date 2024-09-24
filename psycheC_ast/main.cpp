#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include <csv-parser/csv.hpp>

// Function to read the content of the .c file for '.c' mode
std::string readFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return "";
    }
    return {(std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()};
}

//return exit code
int constructAST(const std::string& sourceCode, const std::string& mode) {
    // Step 1: Use a fixed file name (this will be overwritten each time)
    const char* tempFileName = "./tmp/tempSourceCode.c";

    // Step 2: Write the source code into the file (this will overwrite the file each time)
    std::ofstream tempFile(tempFileName);
    if (!tempFile.is_open()) {
        std::cerr << "Failed to open temporary file for writing." << std::endl;
        return 1;
    }
    tempFile << sourceCode;
    tempFile.close();  // Close the file after writing

    // Step 3: Run the external command using system()
    std::string command;
    if (mode == ".csv") {
        command = "(./psychec/cnip -l C -d " + std::string(tempFileName) + ") >/dev/null 2>/dev/null";
    } else {
        command = "./psychec/cnip -l C -d " + std::string(tempFileName);
    }
        return system(command.c_str());

}

void run(const std::string& filePath, std::vector<std::string>& pathsVec, std::string& mode) {
    if (mode == ".csv") {
        csv::CSVFormat format;
        format.delimiter(',')
              .quote('"')
              .header_row(0);
        csv::CSVReader reader(filePath, format);

        long int row_index = 0;

        // Open the row_index.log file (overwrites the file at each run)
        std::ofstream logFile("row_index.log");
        if (!logFile.is_open()) {
            std::cerr << "Error: Could not open row_index.log for writing." << std::endl;
            return;
        }

        // Iterate over the rows of the CSV file
        for (const csv::CSVRow& row : reader) {

            // Get the last column (source code)
            const auto sourceCode = row["flines"].get<std::string>();

            // Construct the AST

            int exit_code = constructAST(sourceCode, mode);
            if (exit_code != 0) {
                // If the AST construction fails, log the row index
                logFile << row_index << std::endl;
            }

            // Analyse the AST (you can add your analysis logic here)

            // Increment the row index
            ++row_index;
        }

        // Signal the end of the file
        std::cout << "EOF" << std::endl;

        // Close the log file
        logFile.close();
    }
    else {
        for (const auto & i : pathsVec) {
            auto source_code = readFile(i);
            int exit_status = constructAST(source_code, mode);
            if (exit_status != 0) {
                std::cerr << "The AST for file:" << i << "was NOT successfully created." << std::endl << std::endl;
            } else {
                std::cout << "The AST for file:" << i << "was INDEED successfully created." << std::endl << std::endl;
            }
            std::cout << "------------------------------------------------------------------------------" << std::endl;
        }
    }
}

int main(const int argc, char* argv[]) {

    // Check if the required arguments are provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << "<mode> <string_path>" << std::endl << "Modes: [.csv, .c]" << std::endl;
        return 1;
    }

    // Convert argv[2] to std::string
    std::string mode = argv[1];
    std::string filePath;
    std::vector<std::string> pathsVec;
    if (mode == ".csv") {
        filePath = argv[2];
    }  else {
        for (int i = 2; i < argc; i++) {
            pathsVec.emplace_back(argv[i]);
        }
    }

    //Running response
    run(filePath, pathsVec, mode);

    return 0;
}
