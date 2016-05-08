import networkx as nx


def calculate_nearest_k_nodes(g, node, k):
    return nx.single_source_shortest_path_length(g, node, k)


def calculate_betweenesss(g):
    return nx.betweenness_centrality(g)


def calculate_degree(g):
    return nx.degree_centrality(g)


def calculate_katz(g):
    return nx.katz_centrality_numpy(g)


def calculate_max_degree_head_tail(g, node):
    predecessors = g.predecessors(node)
    successors = g.successors(node)

    try:
        max_p = max([g.node[p]['degree'] for p in predecessors])
    except ValueError:
        max_p = -1
    try:
        max_s = max([g.node[s]['degree'] for s in successors])
    except ValueError:
        max_s = -1

    max_p_s = max([max_p, max_s])

    return max_p_s


def calculate_min_degree_head_tail(g, node):
    predecessors = g.predecessors(node)
    successors = g.successors(node)

    try:
        max_p = min([g.node[p]['degree'] for p in predecessors])
    except ValueError:
        max_p = 999
    try:
        max_s = min([g.node[s]['degree'] for s in successors])
    except ValueError:
        max_s = 999

    min_p_s = min([max_p, max_s])

    return min_p_s


def calculate_avg_degree_head_tail(g, node):
    predecessors = g.predecessors(node)
    successors = g.successors(node)

    try:
        p_degrees = [g.node[p]['degree'] for p in predecessors]
        sum_p = sum(p_degrees)
        len_p = len(p_degrees)
    except ValueError:
        sum_p = 0
        len_p = 0
    try:
        s_degrees = [g.node[s]['degree'] for s in successors]
        sum_s = sum(s_degrees)
        len_s = len(s_degrees)
    except ValueError:
        sum_s = 0
        len_s = 0

    avg_p_s = sum([sum_p, sum_s]) / sum([len_p, len_s])

    return avg_p_s
