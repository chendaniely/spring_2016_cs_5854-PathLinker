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
        filename = "{}_part_{}_of_{}.tsv".format(base_file_path,
                                                 idx + 1,
                                                 num_parts)
        df = data_shuffled[i:i + len_per_part]
        df.to_csv(filename, index=False, sep='\t')
        cv_data.append(df)
    return cv_data


def generate_parameters():
    transformation = collections.namedtuple('transformation_params',
                                            'dataset fixed_pct value')

    edges_not_in = (
        transformation('not_in', 'fixed', 0.10),
        transformation('not_in', 'pct', 0.25),
        transformation('not_in', 'pct', 0.50),
        transformation('not_in', 'pct', 0.75)
    )

    edges_in = (
        transformation('in', 'pct', 150),
        transformation('in', 'pct', 200),
        transformation('in', 'pct', 300),
        transformation('in', 'fixed', 0.9)

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


def transform_weights(df, param_not_in, param_in,
                      # interactome_weight_col_name='edge_weight',
                      # pathway_weight_col_name='weight',
                      minimum_weight=0,
                      maximum_weight=0.9):
    print("NOT IN before")
    print(df.ix[pd.isnull(df['weight']), 'edge_weight'].head())
    # not_in
    if param_not_in.fixed_pct == 'fixed':
        df.ix[pd.isnull(df['weight']), 'edge_weight'] = param_not_in.value
    elif param_not_in.fixed_pct == 'pct':
        df.ix[pd.isnull(df['weight']), 'edge_weight'] = \
            df.ix[pd.isnull(df['weight']), 'edge_weight'].\
            apply(calculate_new_weight,
                  mult_factor=param_not_in.value,
                  min_value=minimum_weight,
                  max_value=maximum_weight)
    else:
        raise ValueError
    print("NOT IN after")
    print(df.ix[pd.isnull(df['weight']), 'edge_weight'].head())

    # in
    print('IN before')
    print(df.ix[pd.notnull(df['weight']), 'edge_weight'].head())
    if param_in.fixed_pct == 'fixed':
        df.ix[pd.notnull(df['weight']), 'edge_weight'] = param_in.value
    elif param_in.fixed_pct == 'pct':
        df.ix[pd.notnull(df['weight']), 'edge_weight'] = \
            df.ix[pd.notnull(df['weight']), 'edge_weight'].\
            apply(calculate_new_weight,
                  mult_factor=param_in.value,
                  min_value=minimum_weight,
                  max_value=maximum_weight)
    else:
        raise ValueError
    print('IN after')
    print(df.ix[pd.notnull(df['weight']), 'edge_weight'].head())

    return df


def main():
    wnt = pd.read_csv('../data/Wnt-edges.txt', delimiter='\t')
    interactome = pd.read_csv(
        '../data/pathlinker-signaling-children-reg-weighted.txt',
        delimiter='\t')

    cv_data = create_cv_data(wnt, 2, '../output/wnt', 42)

    parameters = generate_parameters()

    all_data = pd.merge(interactome, wnt,
                        on=['#tail', 'head'],
                        how='left')

    for idx, param in enumerate(parameters):
        param_not_in, param_in = param
        print(param_not_in)
        print(param_in)
        df = all_data.copy()
        df = transform_weights(df, param_not_in, param_in)
        filename = "../output/data_ni-{}_i-{}.tsv".format(
            param_not_in.value, param_in.value)
        df.to_csv(filename, index=False, sep='\t')


if __name__ == '__main__':
    main()
