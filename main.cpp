#include <iostream>
#include <fstream>
#include <string>

#include <SyntaxTree.h>
#include <compilation/Compilation.h>
#include <parser/ParseOptions.h>
#include <syntax/SyntaxNode.h>  // Assuming this is where SyntaxNode is defined

// Function to read the content of the .c file
std::string readFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return "";
    }
    return {(std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()};
}

// Recursive DFS function to traverse and print each node
void dfsTraverse(const psy::C::SyntaxNode* node, int depth = 0) {
    if (!node) return;

    // Print the kind of the node, indented based on depth
    std::cout << std::string(depth * 2, ' ') << "Node Kind: " << node->kindText() << std::endl;

    // Recursively visit each child node
    for (const auto* child : node->) {
        dfsTraverse(child, depth + 1);
    }
}

int main() {
    // Path to the C file to be parsed
    const std::string filePath = "ast_test.c";

    // Step 1: Read the file content
    std::string sourceCode = readFile(filePath);
    if (sourceCode.empty()) {
        return 1; // Exit if the file could not be read
    }

    // Step 2: Set up parsing options
    psy::C::ParseOptions parseOpts;
    parseOpts.setAmbiguityMode(psy::C::ParseOptions::AmbiguityMode::DisambiguateAlgorithmically); // Adjust as necessary

    // Step 3: Parse the file content
    auto syntaxTree = psy::C::SyntaxTree::parseText(
        sourceCode,                          // The content of the C file
        psy::C::TextPreprocessingState::Preprocessed, // You can set Preprocessed or Raw depending on the state of the text
        psy::C::TextCompleteness::Fragment,           // Indicate whether the input is a full translation unit or a fragment
        parseOpts,                            // Parse options
        filePath                              // File name (used for reference or error reporting)
    );

    // Step 4 (Optional): Analyze the resulting syntax tree (if necessary)
    // auto compilation = psy::C::Compilation::create("code-analysis");
    // compilation->addSyntaxTree(syntaxTree.get());

    // Step 5: Get the root node of the syntax tree
    auto rootNode = syntaxTree->translationUnitRoot();

    // Step 6: Traverse the syntax tree and print the node kinds
    std::cout << "Traversing the syntax tree in DFS order:" << std::endl;
    dfsTraverse(rootNode);  // Start DFS traversal

    std::cout << "Parsing complete for file: " << filePath << std::endl;

    return 0;
}
