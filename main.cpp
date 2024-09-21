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
bool endsWithC(const std::string& filename) {
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

ASTConstructStatus constructAST(const std::string& sourceCode, const std::string& filePath) {

    // Step 2: Set up parsing options
    psy::C::ParseOptions parseOpts;
    parseOpts.setAmbiguityMode(psy::C::ParseOptions::AmbiguityMode::DisambiguateAlgorithmically); // Adjust as necessary

    // Step 3: Parse the file content
    const auto syntaxTree = psy::C::SyntaxTree::parseText(
        sourceCode,                                   // The content of the C file
        psy::C::TextPreprocessingState::Preprocessed,       // You can set Preprocessed or Raw depending on the state of the text
        psy::C::TextCompleteness::Fragment,             // Indicate whether the input is a full translation unit or a fragment
        parseOpts,                                      // Parse options
        filePath                                        //File name (used for reference or error reporting)
    );
    incrementGeneralCounter();

    if (const auto diagnostics = syntaxTree->diagnostics(); !diagnostics.empty()) {
        if (diagnostics.front().severity() == psy::DiagnosticSeverity::Error)
            return ASTConstructStatus::Error;
        //for now ignore warning
    }

    return ASTConstructStatus::Success;
}

void runAllConstructionAndAnalysis(const std::string& filePath) {
    csv::CSVFormat format;
    format.delimiter(',')
          .quote('"')
          .header_row(0);
    csv::CSVReader reader(filePath, format);

    // Iterate over the rows of the CSV file
    int row_count = 0;
    for (const csv::CSVRow& row : reader) {
        if (endsWithC(row["file"].get<std::string>())) {
            const auto sourceCode = row["flines"].get<std::string>();  // Get the last column (source code)

            constructAST(sourceCode, filePath);

            if (constructStatus != ASTConstructStatus::Error) {
                try analyseAST(syntaxTree);
                except(successful analysis) {
                    incrementSuccessCounter();
                }
            }

            ++row_count;
            if (row_count == 3) break;
        }
    }
}

int main(const int argc, char* argv[]) {

    // Check if there is at least one argument (excluding the program name)
    if (argc > 1) {
        const std::string filePath = argv[1];

        std::cout << "Construction for file: " << filePath << "started." << std::endl;
        runAllConstructionAndAnalysis(filePath);
    }

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
