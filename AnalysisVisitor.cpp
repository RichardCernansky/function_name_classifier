//
// Created by Richard Čerňanský on 20/09/2024.
//
#include <SyntaxTree.h>
#include <syntax/SyntaxNode.h>

#include "AnalysisVisitor.h"

// Constructor definition
AnalysisVisitor::AnalysisVisitor(const psy::C::SyntaxTree* tree)
    : SyntaxVisitor(tree)
{}

// Method to start visiting the syntax tree from the root
void AnalysisVisitor::run(const psy::C::SyntaxNode* root) {
    if (root) {
        visit(root);  // Start visiting from the root node
    }
}

// Override preVisit to print node kinds
bool AnalysisVisitor::preVisit(const psy::C::SyntaxNode *node) {
    if (node) {
        std::cout << "Visiting node-kind: " << to_string(node->kind()) << std::endl;
        return true;
    }
    return false;
}

