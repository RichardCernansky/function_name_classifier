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

#for all the functions in functions.ndjson
    # load one function into tree
    # get all the leaf-to-leaf paths
    # train
