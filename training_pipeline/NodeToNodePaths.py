#file that is primarily in jupyter notebook here is just developed
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__))) # looks for modules also in the parent dir if called from another directory

from extract_functions.Node import Node
import ndjson

def json_to_tree(data: dict) -> Node:
    """
    Recursively builds a tree of Node objects from a JSON dictionary.
    """
    node = Node(
        b_i=None,
        kind=data.get('kind'),
        code_pos=data.get('code_pos'),
        data=data.get('data')
    )

    # Recursively add children
    for child_data in data.get('children', []):
        child_node = json_to_tree(child_data)
        child_node.set_parent(node)  # Set the parent for the child node
        node.add_child(child_node)

    return node

#NODE TO NODE PATHS
# Function to collect all leaf nodes iteratively using DFS
def collect_leaves_iterative(root):
    if root is None:
        return []

    stack = [(root, [])]  # Stack to store (node, path_from_root)
    leaves = []  # List to store leaf nodes and their paths

    while stack:
        node, path = stack.pop()
        current_path = path + [node.kind]  # Update the current path

        # leaf node - has no children
        if not node.children:
            leaves.append((node, current_path))

        # push the children to the stack for DFS
        children = reversed(node.children)
        for child in children:  # process children in order on the stack
            stack.append((child, current_path))

    return leaves

# Function to find the Lowest Common Ancestor (LCA) iteratively
def find_lca_iterative(n1_path, n2_path):
    length = len(n1_path) if len(n1_path) < len(n2_path) else len(n2_path)

    lca = None
    for i in range(length):
        if n1_path[i] == n2_path[i]:
            lca = n1_path[i]
        else:
            break
    return lca

def find_leaf_to_leaf_paths_iterative(root):
    leaf_nodes = collect_leaves_iterative(root)

    # list of all leaf-to-leaf paths
    leaf_to_leaf_paths = []

    # Iterate over each pair of leaf nodes
    for i in range(len(leaf_nodes)):
        for j in range(i + 1, len(leaf_nodes)):
            leaf1, path1 = leaf_nodes[i]
            leaf2, path2 = leaf_nodes[j]

            # find lca
            lca = find_lca_iterative(path1, path2)

            # find the indexes
            lca_index1 = path1.index(lca)
            lca_index2 = path2.index(lca)

            # Path from leaf1 to leaf2 via the LCA
            path_to_lca_from_leaf1 = path1[:lca_index1 + 1]
            path_to_lca_from_leaf2 = path2[:lca_index2 + 1]
            path_to_lca_from_leaf2.reverse()

            # combine the paths
            complete_path = path_to_lca_from_leaf1 + path_to_lca_from_leaf2[1:]

            # Add the complete leaf-to-leaf path to the result
            leaf_to_leaf_paths.append((leaf1.data,) + tuple(complete_path) + (leaf2.data,))

    return [node.data for node, path in leaf_nodes], leaf_to_leaf_paths


def pre_order_traversal(root):
    traversal = []
    def visit(node):
        traversal.append(node.kind)
        for node in node.children:
             visit(node)
    visit(root)
    return traversal

def in_order_traversal(root):
    traversal = []

    def visit(node):
        mid = len(node.children) // 2  
        
        for child in node.children[:mid]:
            visit(child)
        
        traversal.append(node.kind)
        
        for child in node.children[mid:]:
            visit(child)

    visit(root)
    return traversal

def post_order_traversal(root):
    traversal = []
    def visit(node):
        for node in node.children:
             visit(node)
        traversal.append(node.kind)
    visit(root)
    return traversal
