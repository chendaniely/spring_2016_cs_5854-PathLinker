import pandas as pd
import networkx as nx

import tqdm

import helper


def setup():
    pathways = ["BDNF",
                "EGFR1",
                "IL1",
                "IL2",
                "IL3",
                "IL6",
                "IL-7",
                "KitReceptor",
                "Leptin",
                "Prolactin",
                "RANKL",
                "TCR",
                "TGF_beta_Receptor",
                "TNFalpha",
                "Wnt"]
    interactome = pd.read_csv(
        '../data/pathlinker-signaling-children-reg-weighted.txt',
        delimiter='\t')
    return pathways, interactome

#
# Begin Script
#

pathways, interactome = setup()
interactome_e2n = helper.convert_edges_to_node(interactome,
                                               'edge_weight',
                                               'interactome_weight')

# pathlinker_weight_attributes = {}
# pagerank_weight_attributes = {}

# interactome_weight_attributes = {}

for pathway_name in tqdm.tqdm(pathways):
    # pathway_df = pd.read_csv('../data/pathways/{}-edges.txt'.
    #                          format(pathway_name),
    #                          delimiter='\t')

    # pathway_df_n2e = helper.convert_edges_to_node(pathway_df)

    # node_features_df = '../output/features_{}'.format(pathway_name)

    # to get which nodes are sources and targets
    pathway_nodes_df = pd.read_csv('../data/pathways/{}-nodes.txt'.
                                   format(pathway_name),
                                   delimiter='\t')
    pathway_tf_nodes = pathway_nodes_df.ix[
        pathway_nodes_df['node_symbol'] == 'tf']
    pathway_receptor_nodes = pathway_nodes_df.ix[
        pathway_nodes_df['node_symbol'] == 'receptor']

    # create the full interactome as graph
    g = nx.from_pandas_dataframe(interactome_e2n, source='#tail',
                                 target='head',
                                 create_using=nx.DiGraph())

    # add super nodes
    g.add_nodes_from(['super_tf', 'super_receptor'])

    # connect super sources
    receptor_list = list(pathway_receptor_nodes["#node"])
    add_receptor_edges = [('super_receptor', n) for n in receptor_list]
    g.add_edges_from(add_receptor_edges)

    # connect super targets
    tf_list = list(pathway_tf_nodes["#node"])
    add_tf_edges = [(n, 'super_tf') for n in tf_list]
    g.add_edges_from(add_tf_edges)

    distance_closest_source = {}  # tf
    distance_closest_target = {}  # receptor

    for node_idx, node in tqdm.tqdm(enumerate(g.nodes())):
        match = helper.EDGE_NODE_PATTERN.match(node)
        if match:
            try:
                p = nx.shortest_path(g, 'super_receptor', node)
                distance_closest_source[node] = len(p)
            except nx.NetworkXNoPath:
                distance_closest_source[node] = None

            try:
                p = nx.shortest_path(g, node, 'super_tf')
                distance_closest_target[node] = len(p)
            except nx.NetworkXNoPath:
                distance_closest_target[node] = None

    nx.set_node_attributes(g, 'dist_closest_source', distance_closest_source)
    nx.set_node_attributes(g, 'dist_closest_target', distance_closest_target)

    node_filename = '../output/features_{}_02.txt'.format(pathway_name)

    with open(node_filename, 'w') as f:
        col_names = ['name', 'dist_closest_source', 'dist_closest_target']
        fstring = ['%s'] * len(col_names)
        fstring = '\t'.join(fstring) + '\n'

        wstring = fstring % tuple(col_names)
        f.write(wstring)

        for node in tqdm.tqdm(g.nodes_iter()):
            match = helper.EDGE_NODE_PATTERN.match(node)
            if match:
                na = g.node[node]
                f.write(fstring % tuple([
                    node,
                    na['dist_closest_source'],
                    na['dist_closest_target']
                ]))
    print("{} created".format(node_filename))
