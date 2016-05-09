import re

import networkx as nx
import pandas as pd

import tqdm

import calculate_node_attributes as cna
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


def main():
    pathways, interactome = setup()

    # I fucked up.
    # this file should not loop over any of the pathways
    # the features here should be calculated on the entire interactome
    # this is why the variable names will be confusing

    # for pathway_name in pathways:
    #     pathway_df = pd.read_csv('../data/pathways/{}-edges.txt'.
    #                          format(pathway_name),
    #                          delimiter='\t')

    pathway_name = 'interactome'
    pathway_df = interactome

    pathway_df_n2e = helper.convert_edges_to_node(
        pathway_df, weight_col='edge_weight')

    g = nx.from_pandas_dataframe(
        pathway_df_n2e, source='#tail', target='head',
        create_using=nx.DiGraph())

    # print('calculating betweenness_attributes')
    # betweenness_attributes = cna.calculate_betweenesss(g)
    # print('finished betweenness_attributes')

    print('calculating degree_attributes')
    degree_attributes = cna.calculate_degree(g)
    print('finished degree_attributes')

    # print('calculating katz_attributes')
    # katz_attributes = cna.calculate_katz(g)
    # print('finished katz_attributes')

    # nx.set_node_attributes(g, 'betweenness', betweenness_attributes)
    nx.set_node_attributes(g, 'degree', degree_attributes)
    # nx.set_node_attributes(g, 'katz', katz_attributes)

    nearest_1_attributes = {}  # c
    nearest_3_attributes = {}  # c
    # nearest_5_attributes = {}
    max_degree_head_tail_attributes = {}  # c
    min_degree_head_tail_attributes = {}  # c
    avg_degree_head_tail_attributes = {}  # c

    edge_node_pattern = re.compile('.*_to_.*')
    print('iterating over nodes')
    for node in tqdm.tqdm(g.nodes_iter()):
        match = edge_node_pattern.match(node)
        if match:
            n1 = cna.calculate_nearest_k_nodes(g, node, 2)
            nearest_1_attributes[node] = n1

            n3 = cna.calculate_nearest_k_nodes(g, node, 4)
            nearest_3_attributes[node] = n3

            # n5 = cna.calculate_nearest_k_nodes(g, node, 5)
            # nearest_5_attributes[node] = n5

            max_d_ht = cna.calculate_max_degree_head_tail(g, node)
            max_degree_head_tail_attributes[node] = max_d_ht

            min_d_ht = cna.calculate_min_degree_head_tail(g, node)
            min_degree_head_tail_attributes[node] = min_d_ht

            avg_d_ht = cna.calculate_avg_degree_head_tail(g, node)
            avg_degree_head_tail_attributes[node] = avg_d_ht

    print('setting node attributes')
    nx.set_node_attributes(g, 'nearest_1',  nearest_1_attributes)
    nx.set_node_attributes(g, 'nearest_3',  nearest_3_attributes)
    # nx.set_node_attributes(g, 'nearest_5',  nearest_5_attributes)
    nx.set_node_attributes(g, 'max_degree_head_tail',
                           max_degree_head_tail_attributes)
    nx.set_node_attributes(g, 'min_degree_head_tail',
                           min_degree_head_tail_attributes)
    nx.set_node_attributes(g, 'avg_degree_head_tail',
                           avg_degree_head_tail_attributes)
    node_filename = '../output/features_{}_01.txt'.format(pathway_name)
    with open(node_filename, 'w') as f:
        col_names = ['name',
                     # 'betweenness',
                     'degree',
                     # 'katz',
                     'nearest_1',
                     'nearest_3',
                     # 'nearest_5',
                     'max_degree_head_tail', 'min_degree_head_tail',
                     'avg_degree_head_tail']
        fstring = ['%s'] * len(col_names)
        fstring = '\t'.join(fstring) + '\n'

        wstring = fstring % tuple(col_names)
        f.write(wstring)

        print('writing node info')
        for node in tqdm.tqdm(g.nodes_iter()):
            match = edge_node_pattern.match(node)
            if match:
                na = g.node[node]
                f.write(fstring % tuple([
                    node,
                    # na['betweenness'],
                    na['degree'],
                    # na['katz'],
                    na['nearest_1'],
                    na['nearest_3'],
                    # na['nearest_5'],
                    na['max_degree_head_tail'], na['min_degree_head_tail'],
                    na['avg_degree_head_tail']
                ]))
    print("{} created".format(node_filename))

if __name__ == '__main__':
    main()
