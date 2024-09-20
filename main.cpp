#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include <csv-parser/csv.hpp>
#include <SyntaxTree.h>
#include <syntax/SyntaxNode.h>

#include "AnalysisVisitor.h"

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

//make enum for error reporting on constructAndAnalyseAST and runAllConstruction

void constructAndAnalyseAST(const std::string& sourceCode, const std::string& filePath) {
    // if (sourceCode.empty()) {
    //     return 1; // Exit if the file could not be read
    // }

    // Step 2: Set up parsing options
    psy::C::ParseOptions parseOpts;
    parseOpts.setAmbiguityMode(psy::C::ParseOptions::AmbiguityMode::DisambiguateAlgorithmically); // Adjust as necessary

    // Step 3: Parse the file content
    std::cout << "Constructing AST..." << std::endl;
    const auto syntaxTree = psy::C::SyntaxTree::parseText(
        sourceCode,                                   // The content of the C file
        psy::C::TextPreprocessingState::Preprocessed,       // You can set Preprocessed or Raw depending on the state of the text
        psy::C::TextCompleteness::Fragment,             // Indicate whether the input is a full translation unit or a fragment
        parseOpts,                                      // Parse options
        filePath                                        //File name (used for reference or error reporting)
    );

    auto diagnostics = syntaxTree->diagnostics();
    std::cout <<  diagnostics.front().severity() << std::endl;

    // Step 5: Get the root node of the syntax tree
    const auto rootNode = syntaxTree->root();

    // Step 6: Create the AnalysisVisitor and traverse the syntax tree
    AnalysisVisitor analyse_visitor(syntaxTree.get());
    analyse_visitor.run(rootNode);
}

void runAllConstruction(const std::string& filePath) {
    csv::CSVFormat format;
    format.delimiter(',')
          .quote('"')
          .header_row(0);
    csv::CSVReader reader(filePath, format);

    // Iterate over the rows of the CSV file
    for (const csv::CSVRow& row : reader) {
        const auto sourceCode = row["source_code"].get<std::string>();  // Get the last column (source code)
        constructAndAnalyseAST(readFile("ast_test_correct.c.txt")/**sourceCode**/, filePath);
        break;
    }
}

int main(const int argc, char* argv[]) {
    // Check if there is at least one argument (excluding the program name)
    if (argc > 1) {
        const std::string filePath = argv[1];
        runAllConstruction(filePath);
    }





}
