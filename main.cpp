#include <fstream>
#include <iostream>
#include <string>

#include <SyntaxTree.h>
#include <syntax/SyntaxNode.h>

#include "AnalysisVisitor.h"





// Function to read the content of the .c file
std::string readFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return "";
    }
    return {(std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()};
}

int main() {
    // Path to the C file to be parsed
    const std::string filePath = "../datasets/codeforces1/1000.csv";


    // Step 1: Read the file content
    const std::string sourceCode = readFile(filePath);
    if (sourceCode.empty()) {
        return 1; // Exit if the file could not be read
    }

    std::cout << "hi";

    // Step 2: Set up parsing options
    psy::C::ParseOptions parseOpts;
    parseOpts.setAmbiguityMode(psy::C::ParseOptions::AmbiguityMode::DisambiguateAlgorithmically); // Adjust as necessary

    // Step 3: Parse the file content
    const auto syntaxTree = psy::C::SyntaxTree::parseText(
        sourceCode,                          // The content of the C file
        psy::C::TextPreprocessingState::Preprocessed, // You can set Preprocessed or Raw depending on the state of the text
        psy::C::TextCompleteness::Fragment,           // Indicate whether the input is a full translation unit or a fragment
        parseOpts,                            // Parse options
        filePath//File name (used for reference or error reporting)
    );

    // Step 5: Get the root node of the syntax tree
    const auto rootNode = syntaxTree->root();

    // Step 6: Create the AnalysisVisitor and traverse the syntax tree
    std::cout << "Traversing the syntax tree:" << std::endl;
    AnalysisVisitor analyse_visitor(syntaxTree.get());
    analyse_visitor.run(rootNode);
    std::cout << "Parsing complete for file: " << filePath << std::endl;

    return 0;
}
