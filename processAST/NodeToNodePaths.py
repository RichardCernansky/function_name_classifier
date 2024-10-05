from Node import Node
import ndjson


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
    length = n1_path.length if n1_path.length < n2_path.length else n2_path.length

    lca = None
    for i in range(length):
        if n1_path[i] == n2_path[i]:
            lca = n1_path[i]
        else:
            break
    return lca


def find_leaf_to_leaf_paths_iterative(root):
    leaf_nodes = collect_leaves_iterative(root)

    #list of all leaf-to-leaf paths
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
            path_to_lca_from_leaf2.reverse().pop()

            #combine the paths
            complete_path = path_to_lca_from_leaf1 + path_to_lca_from_leaf2

            # Add the complete leaf-to-leaf path to the result
            leaf_to_leaf_paths.append(complete_path)

    return leaf_to_leaf_paths


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


def find_tag(root) -> str:
    for child in root.children:
        if child.kind == "FunctionDefinition":
            definition_node = child
            for definition_child in definition_node.children:
                if definition_child.kind == "FunctionDeclarator":
                    declarator_node = definition_child
                    for declarator_child in declarator_node.children:
                        if declarator_child.kind == "IdentifierDeclarator":
                            return str(declarator_child.data)


def generate_vocabs(file_path):
    # Open the .ndjson file
    with open(file_path, 'r') as ndjson_file:
        # Load the file content
        data = ndjson.load(ndjson_file)

        value_vocab = set()  # set of all leaf values
        path_vocab = set()  # set of all distinct paths
        tags_vocab = set()  # set of all distinct function tags

        for function_json in data:
            # convert each line (function) to a tree
            func_root = json_to_tree(function_json)
            tag = find_tag(func_root)
            func_values, func_paths = find_leaf_to_leaf_paths_iterative(func_root)

            # Update vocabularies
            value_vocab.update(func_values)  # Add function's values to value_vocab set
            path_vocab.update(func_paths)  # Add function's paths to path_vocab set
            tags_vocab.add(tag)  # Add function's tag to tags_vocab set

            # add to vocabs new values from calling find_leafs_to_leaves

        return value_vocab, path_vocab, tags_vocab


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

    #list of all leaf-to-leaf paths
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

            #combine the paths
            complete_path = path_to_lca_from_leaf1 + path_to_lca_from_leaf2[1:]

            # Add the complete leaf-to-leaf path to the result
            leaf_to_leaf_paths.append(complete_path)

    return [node.data for node,path in leaf_nodes], leaf_to_leaf_paths


def main():
    value_vocab, path_vocab, tags_vocab = generate_vocabs('functionsASTs.ndjson')

    #vocab sizes and embedding dimensions
    value_vocab_size = len(value_vocab)
    path_vocab_size = len(path_vocab)
    tags_vocab_size = len(tags_vocab)
    y = tags_vocab_size
    embedding_dim = 128

    print("--------------------DONE--------------------")

if __name__ == "__main__":
    main()