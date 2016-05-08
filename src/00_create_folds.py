"""Even though there is a paramter called `k_folds`
the code in this script hard codes k_folds=2 in various places
"""
import itertools
import collections

import pandas as pd
import networkx as nx
import numpy as np


def setup():
    k_folds = 2
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
    return k_folds, pathways, interactome


def generate_parameters():
    transformation = collections.namedtuple('transformation_params',
                                            'dataset fixed_pct value')

    edges_not_in = (
        # transformation('not_in', 'fixed', 0.10),
        # transformation('not_in', 'pct', 0.25),
        # transformation('not_in', 'pct', 0.50),
        # transformation('not_in', 'pct', 0.75),
        transformation('not_in', 'pct', 1.00),
        # transformation('not_in', 'fixed', 0.0)
    )

    edges_in = (
        # transformation('in', 'pct', 1.50),
        # transformation('in', 'pct', 2.00),
        # transformation('in', 'pct', 3.00),
        transformation('in', 'pct', 1.00),
        # transformation('in', 'fixed', 0.9),
        # transformation('in', 'fixed', 1.0)
    )

    return itertools.product(edges_not_in, edges_in)


def create_reweights(parameters, k_folds, single_pathway, interactome):
    for param_idx, param in enumerate(parameters):
        param_not_in, param_in = param

        for fold_num in range(k_folds):
            fold_num_name = fold_num + 1
            data_file_for_k = pd.read_csv('../output/{}_part_{}_of_{}.txt'.
                                          format(single_pathway,
                                                 fold_num_name,
                                                 k_folds),
                                          delimiter='\t')

            fold_df = pd.merge(interactome, data_file_for_k,
                               on=['#tail', 'head'],
                               how='left')

            transformed_df = transform_weights_io(fold_df,
                                                  param_not_in, param_in,
                                                  single_pathway,
                                                  fold_num_name, k_folds)
            filename = "../output/{}_data_{}_ni-{}-{}_i-{}-{}.txt".format(
                single_pathway,
                fold_num_name,
                param_not_in.fixed_pct, param_not_in.value,
                param_in.fixed_pct, param_in.value)
            transformed_df.to_csv(filename, index=False, sep='\t')
            print("{} created.".format(filename))


def calculate_new_weight(original_value, mult_factor, min_value, max_value):
    new_value = original_value * mult_factor

    if pd.isnull(original_value):
        return np.NaN
    elif new_value > max_value:
        return max_value
    elif new_value < min_value:
        return new_value
    else:
        return new_value


def transform_weights_io(df, param_not_in, param_in,
                         single_pathway, part_number, part_number_total,
                         minimum_weight=0, maximum_weight=0.9):
    # not in
    if param_not_in.fixed_pct == 'fixed':
        df.ix[pd.isnull(df['weight']), 'edge_weight'] = param_not_in.value
    elif param_not_in.fixed_pct == 'pct':
        df.ix[pd.isnull(df['weight']), 'edge_weight'] = \
            df.ix[pd.isnull(df['weight']), 'edge_weight'].\
            apply(calculate_new_weight,
                  mult_factor=param_not_in.value,
                  min_value=minimum_weight,
                  max_value=maximum_weight)
    # in
    if param_in.fixed_pct == 'fixed':
        df.ix[pd.notnull(df['weight']), 'edge_weight'] = param_in.value
    elif param_in.fixed_pct == 'pct':
        df.ix[pd.notnull(df['weight']), 'edge_weight'] = \
            df.ix[pd.notnull(df['weight']), 'edge_weight'].\
            apply(calculate_new_weight,
                  mult_factor=param_not_in.value,
                  min_value=minimum_weight,
                  max_value=maximum_weight)
    return df


def graph_from_pathway(pathway):
    edge_list = pathway[['#tail', 'head']]
    g = nx.from_pandas_dataframe(edge_list, '#tail', 'head',
                                 create_using=nx.DiGraph())
    return(g)


def create_folds(pathway, single_pathway, k_folds, g):
    fold1 = []
    fold2 = []
    for edge in g.edges():
        if (edge[1], edge[0]) in fold1 or (edge[0], edge[1]) in fold1:
            fold1.append(edge)
            continue
        if (edge[1], edge[0]) in fold2 or (edge[0], edge[1]) in fold2:
            fold2.append(edge)
            continue
        if len(fold1) <= len(fold2):
            fold1.append(edge)
        else:
            fold2.append(edge)

    assert set(fold1) != set(fold2)

    for edge in fold1:
        assert set(edge) not in fold2
    for edge in fold2:
        assert set(edge) not in fold1

    for fold_idx, fold in enumerate([fold1, fold2]):
        part_number = fold_idx + 1
        filename = '../output/{}_part_{}_of_{}.txt'.format(
            single_pathway, part_number, k_folds)

        df_edges = pd.DataFrame(fold)
        df_fold = pd.merge(pathway, df_edges,
                           left_on=['#tail', 'head'],
                           right_on=[0, 1])
        df_fold['cv_part'] = fold_idx
        df_fold.to_csv(filename, index=False, sep='\t')
    return fold1, fold2


def create_holdout_nodes(single_pathway, pathway_nodes, fold1, fold2, g):
    f1h = open('../output/nodes_{}_fold1_holdout_nodes.txt'.
               format(single_pathway), 'w')
    f2h = open('../output/nodes_{}_fold2_holdout_nodes.txt'.
               format(single_pathway), 'w')
    f1h.write('#node\tnode_symbol\tnode_type\n')
    f2h.write('#node\tnode_symbol\tnode_type\n')

    nodes = []
    fold1_nodes = []
    fold2_nodes = []
    fold1_holdout_nodes = []
    fold2_holdout_nodes = []
    for edge in g.edges():
        if edge[0] not in nodes:
            nodes.append(edge[0])
        if edge[1] not in nodes:
            nodes.append(edge[1])
    for edge in fold1:
        fold1_nodes.append(edge[0])
        fold1_nodes.append(edge[1])
    for edge in fold2:
        fold2_nodes.append(edge[0])
        fold2_nodes.append(edge[1])
    for node in nodes:
        if node not in fold1_nodes:
            fold1_holdout_nodes.append(node)
        if node not in fold2_nodes:
            fold2_holdout_nodes.append(node)

    for line in pathway_nodes:
        row = line.split('\t')
        if row[0] in fold1_holdout_nodes:
            f1h.write(line)
        if row[0] in fold2_holdout_nodes:
            f2h.write(line)
    f1h.close()
    f2h.close()


def main():
    k_folds, pathways, interactome = setup()

    for single_pathway_idx, single_pathway in enumerate(pathways):
        pathway = pd.read_csv('../data/pathways/{}-edges.txt'.
                              format(single_pathway),
                              delimiter='\t')
        pathway_nodes = open('../data/pathways/{}-nodes.txt'.
                             format(single_pathway))

        g = graph_from_pathway(pathway)

        fold1, fold2 = create_folds(pathway, single_pathway, k_folds, g)

        create_holdout_nodes(single_pathway, pathway_nodes, fold1, fold2, g)

        parameters = generate_parameters()
        create_reweights(parameters, k_folds, single_pathway, interactome)

if __name__ == '__main__':
    main()
