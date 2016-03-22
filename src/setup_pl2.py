#! /usr/env/python

import collections
import itertools

import pandas as pd
import numpy as np


def create_cv_data(data, num_parts, base_file_path, seed=None):
    """
    For a given dataframe, it will be parsed into num_parts with a filename
    based off the base_file_path as a tsv
    """
    len_per_part = round(len(data) / num_parts + 1)
    if seed is not None:
        np.random.seed(seed)
    data_shuffled = data.iloc[np.random.permutation(len(data))]
    cv_data = []
    for idx, i in enumerate(range(0, len(data), len_per_part)):
        filename = "{}_part_{}_of_{}.txt".format(base_file_path,
                                                 idx + 1,
                                                 num_parts)
        df = data_shuffled[i:i + len_per_part]
        df['cv_part'] = idx
        df.to_csv(filename, index=False, sep='\t')
        cv_data.append(df)
    return cv_data


def generate_parameters():
    transformation = collections.namedtuple('transformation_params',
                                            'dataset fixed_pct value')

    edges_not_in = (
        # transformation('not_in', 'fixed', 0.10),
        transformation('not_in', 'pct', 0.25),
        # transformation('not_in', 'pct', 0.50),
        # transformation('not_in', 'pct', 0.75),
        transformation('not_in', 'pct', 1.00),
        transformation('not_in', 'fixed', 0.0)
    )

    edges_in = (
        # transformation('in', 'pct', 1.50),
        # transformation('in', 'pct', 2.00),
        # transformation('in', 'pct', 3.00),
        transformation('in', 'pct', 1.00),
        transformation('in', 'fixed', 0.9),
        transformation('in', 'fixed', 1.0)
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
        print(df.ix[pd.isnull(df['weight']), ['edge_weight', 'cv_part']].head())
        print(df.ix[pd.notnull(df['weight']), ['edge_weight', 'cv_part']].head())

        # not_in
        if param_not_in.fixed_pct == 'fixed':
            df.ix[(pd.isnull(df['weight']) | (df['cv_part'] != cv_part)),
                  'edge_weight'] = param_not_in.value

        elif param_not_in.fixed_pct == 'pct':
            df.ix[(pd.isnull(df['weight']) | (df['cv_part'] != cv_part)),
                  'edge_weight'] = \
                   df.ix[(pd.isnull(df['weight']) | (df['cv_part'] != cv_part)),
                         'edge_weight'].\
                   apply(calculate_new_weight,
                         mult_factor=param_not_in.value,
                         min_value=minimum_weight,
                         max_value=maximum_weight)
        else:
            raise ValueError
        print("NOT IN after")
        print(df.ix[pd.isnull(df['weight']), ['edge_weight', 'cv_part']].head())
        print(df.ix[pd.notnull(df['weight']), ['edge_weight', 'cv_part']].head())

        # in
        print('IN before')
        print(df.ix[pd.notnull(df['weight']), ['edge_weight', 'cv_part']].head())
        # if the weight is not null and if
        if param_in.fixed_pct == 'fixed':
            df.ix[(pd.notnull(df['weight'])) & (df['cv_part'] == cv_part),
            'edge_weight'] = param_in.value
        elif param_in.fixed_pct == 'pct':
            df.ix[(pd.notnull(df['weight'])) & (df['cv_part'] == cv_part),
                  'edge_weight'] = \
                df.ix[(pd.notnull(df['weight'])) & (df['cv_part'] == cv_part),
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
    wnt = pd.read_csv('../data/Wnt-edges.txt', delimiter='\t')
    interactome = pd.read_csv(
        '../data/pathlinker-signaling-children-reg-weighted.txt',
        delimiter='\t')

    cv_data = create_cv_data(data=wnt, num_parts=2,
                             base_file_path='../output/wnt',
                             seed=42)

    wnt_cv_stacked = pd.concat(cv_data)

    all_data = pd.merge(interactome, wnt_cv_stacked,
                        on=['#tail', 'head'],
                        how='left')

    parameters = generate_parameters()

    for idx, param in enumerate(parameters):
        param_not_in, param_in = param
        print(param_not_in)
        print(param_in)
        df_set = transform_weights(all_data, param_not_in, param_in)
        for df_idx, df in enumerate(df_set):
            filename = "../output/data_{}_ni-{}-{}_i-{}-{}.txt".format(
                df_idx,
                param_not_in.fixed_pct, param_not_in.value,
                param_in.fixed_pct, param_in.value)
            df.to_csv(filename, index=False, sep='\t')


if __name__ == '__main__':
    main()
