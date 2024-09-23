//
// Created by Richard Čerňanský on 20/09/2024.
//
#ifndef ANALYSIS_VISITOR_H
#define ANALYSIS_VISITOR_H

#include <syntax/SyntaxVisitor.h>

class AnalysisVisitor final : public psy::C::SyntaxVisitor {
public:
    explicit AnalysisVisitor(const psy::C::SyntaxTree* tree);

    // Method to start visiting the syntax tree from the root
    void run(const psy::C::SyntaxNode* root);

    // Override preVisit to print node kinds
    bool preVisit(const psy::C::SyntaxNode *node) override;
};

#endif // ANALYSIS_VISITOR_H

