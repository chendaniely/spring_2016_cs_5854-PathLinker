import networkx as nx
import pandas as pd

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

    for pathway_name in pathways:
        pathway_df = pd.read_csv('../data/pathways/{}-edges.txt'.
                                 format(pathway_name),
                                 delimiter='\t')

        pathway_df_n2e = helper.convert_edges_to_node(pathway_df)

        g = nx.from_pandas_dataframe(
            pathway_df_n2e, source='#tail', target='head',
            create_using=nx.DiGraph())

        betweenness_attributes = cna.calculate_betweenesss(g)
        degree_attributes = cna.calculate_degree(g)
        katz_attributes = cna.calculate_katz(g)

        nx.set_node_attributes(g, 'betweenness', betweenness_attributes)
        nx.set_node_attributes(g, 'degree', degree_attributes)
        nx.set_node_attributes(g, 'katz', katz_attributes)

        nearest_2_attributes = {}  # c
        nearest_4_attributes = {}  # c
        nearest_6_attributes = {}  # c
        max_degree_head_tail_attributes = {}  # c
        min_degree_head_tail_attributes = {}  # c
        avg_degree_head_tail_attributes = {}  # c

        for node in g.nodes_iter():
            n2 = cna.calculate_nearest_k_nodes(g, node, 2)
            nearest_2_attributes[node] = n2

            n4 = cna.calculate_nearest_k_nodes(g, node, 4)
            nearest_4_attributes[node] = n4

            n6 = cna.calculate_nearest_k_nodes(g, node, 6)
            nearest_6_attributes[node] = n6

            max_d_ht = cna.calculate_max_degree_head_tail(g, node)
            max_degree_head_tail_attributes[node] = max_d_ht

            min_d_ht = cna.calculate_min_degree_head_tail(g, node)
            min_degree_head_tail_attributes[node] = min_d_ht

            avg_d_ht = cna.calculate_avg_degree_head_tail(g, node)
            avg_degree_head_tail_attributes[node] = avg_d_ht

        nx.set_node_attributes(g, 'nearest_2',  nearest_2_attributes)
        nx.set_node_attributes(g, 'nearest_4',  nearest_2_attributes)
        nx.set_node_attributes(g, 'nearest_6',  nearest_2_attributes)
        nx.set_node_attributes(g, 'max_degree_head_tail',
                               max_degree_head_tail_attributes)
        nx.set_node_attributes(g, 'min_degree_head_tail',
                               min_degree_head_tail_attributes)
        nx.set_node_attributes(g, 'avg_degree_head_tail',
                               avg_degree_head_tail_attributes)
        node_filename = '../output/features_{}_01.txt'.format(pathway_name)
        with open(node_filename, 'w') as f:
            col_names = ['name', 'betweenness', 'degree', 'katz',
                         'nearest_2', 'nearest_4', 'nearest_6',
                         'max_degree_head_tail', 'min_degree_head_tail',
                         'avg_degree_head_tail']
            fstring = ['%s'] * len(col_names)
            fstring = '\t'.join(fstring) + '\n'

            wstring = fstring % tuple(col_names)
            f.write(wstring)

            for node in g.nodes_iter():
                na = g.node[node]
                f.write(fstring % tuple([
                    node, na['betweenness'],  na['degree'], na['katz'],
                    na['nearest_2'], na['nearest_4'], na['nearest_6'],
                    na['max_degree_head_tail'], na['min_degree_head_tail'],
                    na['avg_degree_head_tail']
                ]))
        print("{} created".format(node_filename))

if __name__ == '__main__':
    main()
