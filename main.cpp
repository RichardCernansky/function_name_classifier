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
int sucess_counter = 0;
int general_counter = 0;

void incrementSuccessCounter() {
    ++sucess_counter;
}
void incrementGeneralCounter() {
    ++general_counter;
}

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

std::pair<ASTConstructStatus, std::unique_ptr<psy::C::SyntaxTree>>
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
    incrementGeneralCounter();

    // Check diagnostics for errors
    if (const auto diagnostics = syntaxTree->diagnostics(); !diagnostics.empty()) {
        if (diagnostics.front().severity() == psy::DiagnosticSeverity::Error) {
            return std::pair<ASTConstructStatus, std::unique_ptr<psy::C::SyntaxTree>>(
                ASTConstructStatus::Error, std::move(syntaxTree));
        }
        // For now, ignore warnings
    }

    return std::pair<ASTConstructStatus, std::unique_ptr<psy::C::SyntaxTree>>(
        ASTConstructStatus::Success, std::move(syntaxTree));
}


void runAllConstructionAndAnalysis(const std::string& filePath) {
    csv::CSVFormat format;
    format.delimiter(',')
          .quote('"')
          .header_row(0);
    csv::CSVReader reader(filePath, format);

    // Iterate over the rows of the CSV file
    int row_index = 0; //debug variable
    for (const csv::CSVRow& row : reader) {
        std::cout << "row_index: " << row_index << std::endl;
        if (isFileC(row["file"].get<std::string>())) {
            const auto sourceCode = row["flines"].get<std::string>();  // Get the last column (source code)

            // if (row_count == 2) {
            //     std::cout << "hello" << std::endl;
            // }
            auto output_construct = constructAST(sourceCode, filePath);
            // Declare variables for unpacking
            ASTConstructStatus status;
            std::unique_ptr<psy::C::SyntaxTree> syntaxTree;

            // Unpack the pair into separate variables
            std::tie(status, syntaxTree) = std::move(output_construct);
            if (status != ASTConstructStatus::Error) {
                analyseAST(syntaxTree);  // This may throw an exception
                incrementSuccessCounter();  // Only increment if no exception is thrown
            }

            ++row_index;
            if (row_index == 3) break;
        }
    }
}

int main(const int argc, char* argv[]) {

    // Check if there is at least one argument (excluding the program name)
    if (argc > 1) {
        const std::string filePath = argv[1];
        //Running response
        std::cout << "Construction for file: " << filePath << "started." << std::endl;
        runAllConstructionAndAnalysis(filePath);
    }


    //Output response
    std::cout << "Number of successful: " << sucess_counter << std::endl;
    std::cout << "Number of general: " << general_counter << std::endl;
    if (general_counter > 0) {
        // Calculate the percentage and cast to double for proper division
        double percentage = (static_cast<double>(sucess_counter) / general_counter) * 100;
        std::cout << "Percentage: " << percentage << "%" << std::endl;
    } else {
        std::cout << "General counter is zero, cannot calculate percentage." << std::endl;
    }

    return 0;
}
