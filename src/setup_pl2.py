#! /usr/env/python

import collections
import itertools
import random

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def find_grouped_edges(edge_data):
    """Takes a dataframe of edges
    returns a dictionary:
    'edges' are tuples of edges and
    'reverse_edges' are tuples of edges that are reverse of edges

    for each row in the dataframe, we subset the columns that rep the edge
    convert the edge into a tuple (so it can be a dict key)
    """
    same_edges = []
    duplicate_edges = {}

    for row in edge_data.iterrows():
        edge = tuple(row[1][['#tail', 'head']].tolist())
        reverse_edge = tuple(row[1][['head', '#tail']].tolist())
        if edge in same_edges:
            print('duplicate edge: {}'.format(edge))
        elif reverse_edge in same_edges:
            print('duplicate edge as reverse: {} -> {}'.format(edge, reverse_edge))
        else:
            same_edges.append(edge)

        if (reverse_edge in same_edges) and \
           (reverse_edge not in duplicate_edges.keys()):
            print('reverse edge found: {} -> {}'.format(edge, reverse_edge))
            duplicate_edges[reverse_edge] = edge

    for edge_idx, edge in enumerate(same_edges):
        assert edge not in same_edges[edge_idx + 1:]
        assert [edge[1], edge[0]] not in same_edges[edge_idx + 1:]
    return {'edges': same_edges, 'reverse_edges':duplicate_edges}


def num_parts_to_len_per_part(total_size, num_parts):
    return (total_size + 1) // num_parts

def split_into_n_parts(edges, size_per_part):
    # this will only work if the size_per_part evenly divides
    # return [list(t) for t in zip(*[iter(edges)] * size_per_part)]

    l = [edges[i:(i + size_per_part)] for i in range(0, len(edges), size_per_part)]
    assert l[0] != l[1]
    return l

def sample_edges_for_fold(grouped_edge_dict, num_parts):
    """Returns a 2d list of edges sampled for folds
    """
    edges = grouped_edge_dict['edges']
    print('length of edges: {}'.format(len(edges)))

    for edge_idx, edge in enumerate(edges):
        assert edge not in edges[edge_idx + 1:]
        assert [edge[1], edge[0]] not in edges[edge_idx + 1:]
    
    random.shuffle(edges)

    size_per_part = num_parts_to_len_per_part(len(edges), num_parts)
    print('size per part: {}'.format(size_per_part))
    
    sampled_edges = split_into_n_parts(edges, size_per_part)
    
    print('len of sampled edges: {}'.format(len(sampled_edges)))

    for edge in sampled_edges[0]:
        assert edge not in sampled_edges[1]
        assert grouped_edge_dict['reverse_edges'][edge] == [edge[1], edge[0]]
        assert grouped_edge_dict['reverse_edges'][edge] not in sampled_edges[1]

    return sampled_edges

def append_reverse_edges(sampled_edges, reverse_edges):
    """sampled_edges is a list where the first level is the folds
    the second level contain tuples of the edges
 
    for each edge in the fold, if a reverse edge exists,
    the reverse edge is added to the fold
    """
    # matched_reverse_edges = []
    # for fold in sampled_edges:
    #     reverse_edge_fold = []
    #     for edge in reverse_edges:
    #         if edge in reverse_edges.keys():
    #             reverse_edge_fold.append(reverse_edges[edge])
    #             assert reverse_edges[edge] not in reverse_edges
    #     matched_reverse_edges.append(reverse_edge_fold)
    # return matched_reverse_edges

    for edge in sampled_edges[0]:
        assert edge not in sampled_edges[1]
        assert reverse_edges[edge] not in sampled_edges[1]
    print(type(reverse_edges))

    print("append reverse edges")
    print("len sampled edges: {}".format(len(sampled_edges)))
    print('len reverse_edges: {}'.format(len(reverse_edges)))

    new_sampled_edges = []
    for fold_idx, fold in enumerate(sampled_edges):
        new_fold_values = []
        # found_reverse_edges = []
        for edge in fold:
            new_fold_values.append(edge)
            if edge in reverse_edges.keys():
                print("found edge in reverse_edges {}, {}. fold: {}".\
                      format(edge, reverse_edges[edge], fold_idx))
                assert edge != reverse_edges[edge]
                new_fold_values.append(reverse_edges[edge])
        new_sampled_edges.append(new_fold_values)
    assert len(new_sampled_edges) == 2
    print("len new sampled edges: {}".format(len(new_sampled_edges)))
    print("len new sampled edges: {}".format(len(new_sampled_edges[0])))
    print("len new sampled edges: {}".format(len(new_sampled_edges[1])))

    for edge in new_sampled_edges[0]:
        assert edge not in new_sampled_edges[1], "duplication error in append_reverse_edges() {}".format(edge)
    return new_sampled_edges

def create_fold_data(data, num_parts, base_file_path, seed=None,
                     edge_from='#tail', edge_to='head'):
    """
    For a given dataframe, it will be parsed into num_parts with a filename
    based off the base_file_path as a tsv
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    print('creating fold data')

    edges_only = data[[edge_from, edge_to]]
    print(edges_only.shape)
    unique_edges = edges_only.drop_duplicates()
    print(unique_edges.shape)
    print(unique_edges.head())
    print(type(unique_edges))

    dict_edges_reverse = find_grouped_edges(unique_edges)
    # print(dict_edges_reverse['reverse_edges'])
    sampled_edges = sample_edges_for_fold(dict_edges_reverse, num_parts)
    print("SAMPLED EDGES")
    print(sampled_edges)
    assert sampled_edges[0] != sampled_edges[1]

    for e in sampled_edges:
        print(len(e))


    for edge in sampled_edges[0]:
        assert edge not in sampled_edges[1]
        assert [edge[1], edge[0]] not in sampled_edges[1]
        assert dict_edges_reverse['reverse_edges'][edge] not in sampled_edges[1]
    
    edges_for_fold = append_reverse_edges(sampled_edges,
                                          dict_edges_reverse['reverse_edges'])
    print("len edges_for_fold: {}".format(len(edges_for_fold)))
    for fold in edges_for_fold:
        print(len(fold))

    # make sure there are no common edges in the folds
    for edge in edges_for_fold[0]:
        assert edge not in edges_for_fold[1], "Folds share common edge:{}".format(edge)
 
    for edges_per_fold_idx, edges_per_fold in enumerate(edges_for_fold):
        print('#'*10)
        print(len(edges_per_fold))
        filename = "{}_part_{}_of_{}.txt".format(base_file_path,
                                                 edges_per_fold_idx + 1,
                                                 num_parts)
        df_edges = pd.DataFrame(edges_per_fold)
        df_fold = pd.merge(data, df_edges,
                           left_on=['#tail', 'head'],
                           right_on=[0, 1])
        print(df_fold.head())
        df_fold['cv_part'] = edges_per_fold_idx
        df_fold.to_csv(filename, index=False, sep='\t')
        print("Fold data created: {}".format(filename))

    # len_per_part = num_parts_to_len_per_part(len(data), num_parts)

    # data_shuffled = data.iloc[np.random.permutation(len(data))]

    # for idx, i in enumerate(range(0, len(data), len_per_part)):
    #     filename = "{}_part_{}_of_{}.txt".format(base_file_path,
    #                                              idx + 1,
    #                                              num_parts)
    #     df = data_shuffled[i:(i + len_per_part)]
    #     df['cv_part'] = idx
    #     df.to_csv(filename, index=False, sep='\t')
    #     print("Fold data created: {}".format(filename))
    #     # cv_data.append(df)
    return None  # cv_data


def generate_parameters():
    transformation = collections.namedtuple('transformation_params',
                                            'dataset fixed_pct value')

    edges_not_in = (
        # transformation('not_in', 'fixed', 0.10),
        transformation('not_in', 'pct', 0.25),
        # transformation('not_in', 'pct', 0.50),
        # transformation('not_in', 'pct', 0.75),
        # transformation('not_in', 'pct', 1.00),
        # transformation('not_in', 'fixed', 0.0)
    )

    edges_in = (
        # transformation('in', 'pct', 1.50),
        # transformation('in', 'pct', 2.00),
        # transformation('in', 'pct', 3.00),
        # transformation('in', 'pct', 1.00),
        transformation('in', 'fixed', 0.9),
        # transformation('in', 'fixed', 1.0)
    )

    return itertools.product(edges_not_in, edges_in)


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


def transform_weights(all_data, param_not_in, param_in,
                      # interactome_weight_col_name='edge_weight',
                      # pathway_weight_col_name='weight',
                      minimum_weight=0,
                      maximum_weight=0.9):
    """
    If the 'weights' column in the full interactome is `nan`,
    then the edge is "not in" the pathway of interest.
    Additionally, for each unique value of cv_parts, if the value does not
    match a cv_part, it will also be considered 'not in'

    When the weights column in the full interactome is not `nan',
    it represents an edge that is part of the pathway of interest.
    we then use the cv_part vlues to set the `in` value
    """
    cv_parts = all_data['cv_part'].unique()
    cv_parts = cv_parts[~np.isnan(cv_parts)]
    print(cv_parts)
    transformed_dfs = []

    for cv_part in cv_parts:
        df = all_data.copy()
        # create a new df based on the cv_parts
        # the rows that are not in cv_parts will be the param_not_in
        # the rows that are in the cv_parts will be the param_in

        # if the weight is null, and if the cv_part is not the current value
        # then consider the row as "not in"
        print("NOT IN before")
        print(df.ix[pd.isnull(df['weight']), [
              'edge_weight', 'cv_part']].head())
        print(df.ix[pd.notnull(df['weight']), [
              'edge_weight', 'cv_part']].head())

        # not_in
        if param_not_in.fixed_pct == 'fixed':
            df.ix[(pd.isnull(df['weight']) | (df['cv_part'] != cv_part)),
                  'edge_weight'] = param_not_in.value

        elif param_not_in.fixed_pct == 'pct':
            df.ix[(pd.isnull(df['weight']) | (df['cv_part'] != cv_part)),
                  'edge_weight'] = df.ix[(pd.isnull(df['weight']) | (df['cv_part'] != cv_part)),
                                         'edge_weight'].\
                apply(calculate_new_weight,
                      mult_factor=param_not_in.value,
                      min_value=minimum_weight,
                      max_value=maximum_weight)
        else:
            raise ValueError
        print("NOT IN after")
        print(df.ix[pd.isnull(df['weight']), [
              'edge_weight', 'cv_part']].head())
        print(df.ix[pd.notnull(df['weight']), [
              'edge_weight', 'cv_part']].head())

        # in
        print('IN before')
        print(df.ix[pd.notnull(df['weight']), [
              'edge_weight', 'cv_part']].head())
        # if the weight is not null and if
        if param_in.fixed_pct == 'fixed':
            df.ix[(pd.notnull(df['weight'])) & (df['cv_part'] == cv_part),
                  'edge_weight'] = param_in.value
        elif param_in.fixed_pct == 'pct':
            df.ix[(pd.notnull(df['weight'])) & (df['cv_part'] == cv_part),
                  'edge_weight'] = df.ix[(pd.notnull(df['weight'])) & (df['cv_part'] == cv_part),
                                         'edge_weight'].\
                apply(calculate_new_weight,
                      mult_factor=param_in.value,
                      min_value=minimum_weight,
                      max_value=maximum_weight)
        else:
            raise ValueError
        print('IN after')
        print(df.ix[pd.notnull(df['weight']), 'edge_weight'].head())
        transformed_dfs.append(df)

    return transformed_dfs


def main():
    pathways = ["BDNF", "EGFR1",
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
    # pathways = ["IL3", "IL6"]

    interactome = pd.read_csv(
        '../data/pathlinker-signaling-children-reg-weighted.txt',
        delimiter='\t')

    k_folds = 2

    # for each pathway
    for single_pathway_idx, single_pathway in enumerate(pathways):
        print("*"*80)
        print("Generating data for pathway: {}".format(single_pathway))
        pathway = pd.read_csv('../data/pathways/{}-edges.txt'.
                              format(single_pathway),
                              delimiter='\t')

        edge_list = pathway[['#tail', 'head']]
        g = nx.from_pandas_dataframe(edge_list, '#tail', 'head',
                                     create_using=nx.DiGraph())
        print('nx EDGES')
        print(g.edges())

        fold1 = []
        fold2 = []
        for edge in g.edges():
            print((edge[0], edge[1]))
            if (edge[1], edge[0]) in fold1 or (edge[0], edge[1]) in fold1:
                fold1.append(edge)
                print('fold 1 dup')
                continue
            if (edge[1], edge[0]) in fold2 or (edge[0], edge[1]) in fold2:
                fold2.append(edge)
                print('fold 2 dup')
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

        print('~'*100)
        print(sorted(fold1))
        print('*'*80)
        print(sorted(fold2))

        pathway_nodes = pd.read_csv('../data/pathways/{}-nodes.txt'.
                                    format(single_pathway),
                                    delimiter='\t')

        pathway_nodes = open('../data/pathways/{}-nodes.txt'.format(single_pathway))
        f1h = open('../output/nodes/{}_fold1_holdout_nodes.txt'.format(single_pathway), 'w')
        f2h = open('../output/nodes/{}_fold2_holdout_nodes.txt'.format(single_pathway), 'w')

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

        print(len(nodes), len(fold1_holdout_nodes), len(fold2_holdout_nodes))

        for line in pathway_nodes:
            row = line.split('\t')
            if row[0] in fold1_holdout_nodes:
                f1h.write(line)
            if row[0] in fold2_holdout_nodes:
                f2h.write(line)
        f1h.close()
        f2h.close()
        # assert False

        # cv_data = create_fold_data(data=pathway, num_parts=k_folds,
        #                            base_file_path='../output/{}'.
        #                            format(single_pathway),
        #                            seed=42)

        cv_data = [fold1, fold2]

        for fold_idx, fold in enumerate(cv_data):
            part_number = fold_idx + 1
            filename = '../output/{}_part_{}_of_{}.txt'.format(
                single_pathway, part_number, k_folds)
            print(filename)
            df_edges = pd.DataFrame(fold)
            df_fold = pd.merge(pathway, df_edges,
                               left_on=['#tail', 'head'],
                               right_on=[0, 1])
            print(df_edges.head())
            print(df_fold.head())
            df_fold['cv_part'] = fold_idx
            df_fold.to_csv(filename, index=False, sep='\t')
            print(df_fold.head())

        # for edges_per_fold_idx, edges_per_fold in enumerate(edges_for_fold):
        #     print('#'*10)
        #     print(len(edges_per_fold))
        #     filename = "{}_part_{}_of_{}.txt".format(
        #                                              edges_per_fold_idx + 1,
        #                                              num_parts)
        #     df_edges = pd.DataFrame(edges_per_fold)
        #     df_fold = pd.merge(data, df_edges,

        #     print(df_fold.head())
        #     df_fold['cv_part'] = edges_per_fold_idx
        #     df_fold.to_csv(filename, index=False, sep='\t')
        #     print("Fold data created: {}".format(filename))

        
        parameters = generate_parameters()
        print("Parameters: {}".format(parameters))

        # for each parameter set
        for param_idx, param in enumerate(parameters):
            param_not_in, param_in = param
            print(param_not_in)
            print(param_in)

            # for each fold
            for fold_num in range(k_folds):
                fold_num_name = fold_num + 1
                data_file_for_k = pd.read_csv('../output/{}_part_{}_of_{}.txt'.
                                              format(single_pathway,
                                                     fold_num_name,
                                                     k_folds),
                                              delimiter='\t')

                print(data_file_for_k.head())
                fold_df = pd.merge(interactome, data_file_for_k,
                                   on=['#tail', 'head'],
                                   how='left')
                print(fold_df['cv_part'].unique())
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
if __name__ == '__main__':
    main()
