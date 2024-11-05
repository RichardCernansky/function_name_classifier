import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Node import Node

class NodeTree:
    def __init__(self, root_node: Node):
        self.root_node = root_node