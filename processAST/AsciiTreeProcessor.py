from Node import Node
from typing import List

class AsciiTreeProcessor:
    def __init__(self, tree: str):
        self.lines = self.remove_empty_back(tree.split("\n")[1:])


    @staticmethod
    def remove_empty_back(lines: List[str]) -> List[str]:
        while lines[len(lines) - 1] == "":
            lines.pop()
        return lines

    def produce_tree(self):
        root_node = Node(-1)
        cur_node = root_node
        line_idx = 1

        while line_idx < len(self.lines):
            b_i = cur_node.branching_idx
            line_b_i = self.lines[line_idx].find('|--')

            if line_b_i > b_i:
                new_node = Node(line_b_i)
                new_node.set_parent(cur_node)
                cur_node.add_child(new_node)
                cur_node = new_node
                line_idx +=1
            elif line_b_i == b_i:
                new_node = Node(line_b_i)
                cur_node.parent.add_child(new_node)
                new_node.set_parent(cur_node.parent)
                cur_node = new_node
                line_idx +=1
            else:
                cur_node = cur_node.parent

        return root_node