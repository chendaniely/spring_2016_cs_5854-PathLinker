import re

import pandas as pd


def convert_edges_to_node(data, weight_col='weight',
                          weight_col_name='weight'):
    """
    Takes a dataframe of edge lists and converts each edge into a node
    """
    data['edge_node'] = data['#tail'].astype(str) + \
        '_to_' + \
        data['head'].astype(str)

    new_edges_1 = data[['#tail', 'edge_node', weight_col]]
    new_edges_2 = data[['edge_node', 'head', weight_col]]

    new_edges_1.columns = ['#tail', 'head', weight_col_name]
    new_edges_2.columns = ['#tail', 'head', weight_col_name]

    new_data = pd.concat([new_edges_1, new_edges_2], ignore_index=True)
    return new_data


def keep_edge_nodes(df_e2n,
                    keep_columns,
                    search_column='head',
                    search_pattern='.*_to_.*',
                    ):
    """subsets a e2n df by the search_column, if there is a search_pattern match
    it will keep that row
    """
    row_subset = df_e2n.ix[df_e2n[search_column].
                           str.match(search_pattern, as_indexer=True)]
    column_subset = row_subset.reset_index()
    column_subset = column_subset[keep_columns]
    return column_subset


def convert_edgelist_to_node(edge_list, new_col_name='n2n',
                             e1_col='#node1', e2_col='node2'):
    """Similar to convert_edges_to_node, but works on a simple edgelist file
    """
    edge_list[new_col_name] = edge_list[e1_col].astype(str) + \
        '_to_' + \
        edge_list[e2_col].astype(str)
    return(edge_list[new_col_name])


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

EDGE_NODE_PATTERN = re.compile('.*_to_.*')
