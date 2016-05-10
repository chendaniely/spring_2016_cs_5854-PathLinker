import pandas as pd
import numpy as np

from tqdm import tqdm

import helper

pathways, interactome = helper.setup()

interactome_e2n = helper.convert_edges_to_node(interactome,
                                               'edge_weight',
                                               'interactome_weight')
interactome_en = helper.keep_edge_nodes(interactome_e2n,
                                        ['head', 'interactome_weight'])

interactome_degrees = pd.read_csv(
    '../output/features_interactome_no_nearest_01.txt', delimiter='\t')

interactome_features = pd.merge(interactome_degrees,
                                interactome_en, left_on='name',
                                right_on='head')
interactome_features.drop('head', axis=1, inplace=True)

num_folds = 2

create_additional_featuers = False

for pathway in tqdm(pathways):
    pathway_dist_score = pd.read_csv(
        '../output/features_{}_03.txt'.format(pathway), delimiter='\t',
        na_values=['None'])

    pathway_dist_score[['dist_closest_source', 'dist_closest_target']] =\
        pathway_dist_score[['dist_closest_source', 'dist_closest_target']].\
        fillna(value=100000)

    pathway_learning_df = pd.merge(interactome_features,
                                   pathway_dist_score, on='name')

    for fold_idx in tqdm(range(num_folds)):
        fold_num = fold_idx + 1

        # assign class value to NA
        pathway_learning_df['class'] = np.NaN

        # read negative edges
        negatives_file = '../data/negagive_samples/fold{}/'\
            '{}-HALF{}-exclude_none-50X-negative-edges.txt'.format(
                fold_num, pathway, fold_num)
        negatives = pd.read_csv(negatives_file, delimiter='\t')
        neg_edges = helper.convert_edgelist_to_node(negatives)

        # read in positive samples
        positives_file = '../data/positive_samples/{}_part_{}_of_2.txt'.\
            format(pathway, fold_num)
        positives = pd.read_csv(positives_file, delimiter='\t')
        pos_edges = helper.convert_edgelist_to_node(positives,
                                                    e1_col='#tail',
                                                    e2_col='head')

        # assign class 0 to negative edges
        pathway_learning_df.ix[
            pathway_learning_df['name'].isin(neg_edges), 'class'] = 0

        # assign class 1 to positive samples
        pathway_learning_df.ix[
            pathway_learning_df['name'].isin(pos_edges), 'class'] = 1

        # get pos neg values for testing
        if fold_num == 1:
            other_file_idx = 2
        else:
            other_file_idx = 1
        other_negatives_file = '../data/negagive_samples/fold{}/'\
            '{}-HALF{}-exclude_none-100000X-negative-edges.txt'.format(
                other_file_idx, pathway, other_file_idx)
        other_negatives = pd.read_csv(other_negatives_file, delimiter='\t')
        other_neg_edges = helper.convert_edgelist_to_node(other_negatives)
        other_positives_file = '../data/positive_samples/'\
                               '{}_part_{}_of_{}.txt'.\
            format(pathway, other_file_idx, num_folds)
        other_positives = pd.read_csv(other_positives_file, delimiter='\t')
        other_pos_edges = helper.convert_edgelist_to_node(other_positives,
                                                          e1_col='#tail',
                                                          e2_col='head')

        # get training and prediction/testing datasets
        prediction_df = pd.DataFrame({
            'name': pd.concat([other_neg_edges, other_pos_edges],
                              ignore_index=True)})
        prediction_df = pd.merge(
            prediction_df, pathway_learning_df, on='name')

        # interactime_train = drop class NA values
        training_set = pathway_learning_df.ix[
            pd.notnull(pathway_learning_df['class'])]

        if create_additional_featuers:
            training_set = helper.create_additional_features(training_set)
            prediction_df = helper.create_additional_features(prediction_df)

        prediction_filename = '../output/fit_prediction_{}_{}_of_{}'.format(
            pathway, fold_num, num_folds)
        training_filename = '../output/fit_training_{}_{}_of_{}'.format(
            pathway, fold_num, num_folds)

        prediction_df.to_csv(prediction_filename, index=False, sep='\t')
        training_set.to_csv(training_filename, index=False, sep='\t')
