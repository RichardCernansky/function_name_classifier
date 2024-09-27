class Node:
    def __init__(self, b_i: int, kind: str, code_pos: str, data: str):
        self.branching_idx = b_i
        self.parent = None
        self.children = []
        self.kind = kind
        self.code_pos = code_pos
        self.data = data

    def set_parent(self, parent: 'Node'):
        self.parent = parent

    def add_child(self, child: 'Node'):
        self.children.append(child)

    def to_dict(self):
        """Convert the node and its children to a dictionary."""
        return {
            'kind': self.kind,
            'code_pos': self.code_pos,
            'data': self.data,
            'children': [child.to_dict() for child in self.children]
        }