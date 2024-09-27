class Node:
    def __init__(self, b_i: int):
        self.branching_idx = b_i
        self.parent = None
        self.children = []
        self.kind = None
        self.code_pos = None
        self.data = None

    def set_parent(self, parent: 'Node'):
        self.parent = parent

    def add_child(self, child: 'Node'):
        self.children.append(child)