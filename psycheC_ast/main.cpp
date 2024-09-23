#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include <csv-parser/csv.hpp>
#include <SyntaxTree.h>
#include <syntax/SyntaxNode.h>

#include "AnalysisVisitor.h"

// Global counter


enum class ASTConstructStatus {
    Success,
    Warning,
    Error
};

//TEMPORARY
// Function to read the content of the .c file
std::string readFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return "";
    }
    return {(std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()};
}

//helper function to get only the rows of .csv dataset that end with '.c'
bool isFileC(const std::string& filename) {
    // Check if the string ends with ".c"
    if (filename.size() >= 2 && filename.rfind(".c") == (filename.size() - 2)) {
        return true;
    }
    return false;
}

void analyseAST(const std::unique_ptr<psy::C::SyntaxTree>& syntaxTree) {
    // Step 5: Get the root node of the syntax tree
    const auto rootNode = syntaxTree->root();

    // Step 6: Create the AnalysisVisitor and traverse the syntax tree
    AnalysisVisitor analyse_visitor(syntaxTree.get());
    analyse_visitor.run(rootNode);
}

std::unique_ptr<psy::C::SyntaxTree>
constructAST(const std::string& sourceCode, const std::string& filePath) {

    // Step 2: Set up parsing options
    psy::C::ParseOptions parseOpts;
    parseOpts.setAmbiguityMode(psy::C::ParseOptions::AmbiguityMode::DisambiguateAlgorithmically); // Adjust as necessary

    // Step 3: Parse the file content
    auto syntaxTree = psy::C::SyntaxTree::parseText(
        sourceCode,                                     // The content of the C file
        psy::C::TextPreprocessingState::Preprocessed,   // Preprocessed or Raw, depending on the state of the text
        psy::C::TextCompleteness::Fragment,             // Full translation unit or fragment
        parseOpts,                                      // Parse options
        filePath                                        // File name (used for reference or error reporting)
    );

    return syntaxTree;
}


void runAllConstructionAndAnalysis(const std::string& filePath, const long int start_row_index) {
    csv::CSVFormat format;
    format.delimiter(',')
          .quote('"')
          .header_row(0);
    csv::CSVReader reader(filePath, format);


    long int row_index = 0;
    // Iterate over the rows of the CSV file
    for (const csv::CSVRow& row : reader) {
        if (row_index < start_row_index) {
            ++row_index;
            continue;
        }

        //log the row_index before segmentation fault
        std::cout << "row_index: " << row_index << std::endl;

        // Get the last column (source code)
        const auto sourceCode = row["flines"].get<std::string>();

        //Construct the tree
        auto syntaxTree = constructAST(sourceCode, filePath);
        //Analyse the tree
        analyseAST(syntaxTree);  // This may throw an exception

        //handle row index
        ++row_index;
    }

    // Signal end of the file
    std::cout << "EOF" << std::endl;
}

int main(const int argc, char* argv[]) {

    // Check if the required arguments are provided
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <csv_file> <start_row_index>" << std::endl;
        return 1;
    }

    // Get the CSV file name and start row index from command-line arguments
    std::string csv_file = argv[1];
    //TODO resolve long int row_index
    const long int start_row_index = std::stoi(argv[2]);

    const std::string filePath = argv[1];
    //Running response
    runAllConstructionAndAnalysis(filePath, start_row_index);

    return 0;
}
