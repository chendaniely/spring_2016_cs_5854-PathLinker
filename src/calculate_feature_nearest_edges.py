import networkx as nx

def calculate_nearest_edgenode(graph, node, distance_from_node):
    current_set = set([node])
    visited_set = set()
    visited_set.update(current_set)
    for m in nx.nodes_iter(graph):
        if m in nx.single_source_shortest_path_length(
                graph, node, cutoff=distance_from_node) and\
           m not in visited_set:
            visited_set.add(m)
    current_set = visited_set
    return current_set

def main():
    


if __name__ == '__main__':
    main()
