import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Node import Node
from typing import List
import copy


class AsciiTreeProcessor:
    def __init__(self, tree: str):
        self.lines = self.remove_empty_back(tree.split("\n")[1:])

    @staticmethod
    def remove_empty_back(lines: List[str]) -> List[str]:
        while lines[len(lines) - 1] == "":
            lines.pop()
        return lines

    @staticmethod
    def process_line(raw_line: str) -> List[str]:
        while True:
            raw_line_before = copy.deepcopy(raw_line)
            raw_line = raw_line.lstrip("|").lstrip("--").lstrip()

            if raw_line == raw_line_before:
                break

        line_split = raw_line.split()

        if len(line_split) != 3:
            first = True if len(line_split[0]) != 0 else False
            second = True if len(line_split) > 1 and line_split[1][0] == '<' else False
            third = True if len(line_split) > 2 and line_split[2][0] == '`' else False

            if not first:
                line_split.insert(0, '')
            if not second:
                line_split.insert(1, '')
            if not third:
                line_split.insert(2, '')

        line_split[2] = line_split[2].replace("`", "")
        return line_split


    def produce_tree(self) -> Node:
        root_node = Node(-1, "TranslationUnit", "", "")
        cur_node = root_node
        line_idx = 1

        while line_idx < len(self.lines):
            b_i = cur_node.branching_idx
            line_b_i = self.lines[line_idx].find('|--')

            if line_b_i > b_i:
                stripped_line = self.process_line(self.lines[line_idx])
                new_node = Node(line_b_i, stripped_line[0], stripped_line[1], stripped_line[2])
                new_node.set_parent(cur_node)
                cur_node.add_child(new_node)
                cur_node = new_node
                line_idx +=1
            elif line_b_i == b_i:
                stripped_line = self.process_line(self.lines[line_idx])
                new_node = Node(line_b_i, stripped_line[0], stripped_line[1], stripped_line[2])
                cur_node.parent.add_child(new_node)
                new_node.set_parent(cur_node.parent)
                cur_node = new_node
                line_idx +=1
            else:
                cur_node = cur_node.parent

        return root_node

    @staticmethod
    def get_num_nodes(func_root):
        # base case
        if len(func_root.children) == 0:
            return 1  # count leaf node as one

        # recursive case
        total_nodes = 1
        for child in func_root.children:
            total_nodes += AsciiTreeProcessor.get_num_nodes(child)

        return total_nodes

    @staticmethod
    def get_ast_depth(func_root):
        if len(func_root.children) == 0:
            return 1

        max_depth = 0
        for child in func_root.children:
            max_depth = max(max_depth, AsciiTreeProcessor.get_ast_depth(child))

        return max_depth + 1


    @staticmethod
    def get_num_tokens(func_root):
        if len(func_root.children) == 0:
            # if leaf node -> count tokens in the value
            return 1 \
                    if hasattr(func_root, "data") and isinstance(func_root.data,str) else 0

        #recursion
        total_tokens = 0
        for child in func_root.children:
            total_tokens += AsciiTreeProcessor.get_num_tokens(child)

        return total_tokens

    @staticmethod
    def concat_leaf_data_dfs(root: Node) -> str:
        if not root.children:
            return root.data if isinstance(root.data, str) else ""

        result = ""

        for child in root.children:
            result += AsciiTreeProcessor.concat_leaf_data_dfs(child)

        return result


