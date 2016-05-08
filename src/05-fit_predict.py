import pandas as pd
import numpy as np
from sklearn import svm

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
    pathways, original_interactome = setup()

    # for each pathway
    for pathway in pathways:

        # for each fold
        for fold_idx in range(2):
            fold_num = fold_idx + 1

            # read in interactome
            interactome_e2n = helper.convert_edges_to_node(
                original_interactome,
                'edge_weight',
                'interactome_weight')
            interactome_edge_nodes = helper.keep_edge_nodes(
                interactome_e2n, ['head', 'interactome_weight'])

            # assign class value to NA
            interactome_edge_nodes['class'] = np.NaN

            # read negative edges
            negatives_file = '../data/negagive_samples/fold{}/'\
                '{}-HALF{}-exclude_none-1X-negative-edges.txt'.format(
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
            interactome_edge_nodes.ix[
                interactome_edge_nodes['head'].isin(neg_edges), 'class'] = 0

            # assign class 1 to positive samples
            interactome_edge_nodes.ix[
                interactome_edge_nodes['head'].isin(pos_edges), 'class'] = 1

            # save class NA to predict set
            # prediction_df = interactome_edge_nodes.ix[
            #     pd.isnull(interactome_edge_nodes['class']),
            #     ['head', 'interactome_weight']]
            # prediction_set = prediction_df.iloc[:, 1:]

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
                                   '{}_part_{}_of_2.txt'.\
                format(pathway, other_file_idx)
            other_positives = pd.read_csv(other_positives_file, delimiter='\t')
            other_pos_edges = helper.convert_edgelist_to_node(other_positives,
                                                              e1_col='#tail',
                                                              e2_col='head')
            prediction_df = pd.DataFrame({
                'head': pd.concat([other_neg_edges, other_pos_edges],
                                  ignore_index=True)})
            prediction_df = pd.merge(
                prediction_df, interactome_edge_nodes, on='head')
            prediction_df = prediction_df[['head', 'interactome_weight']]
            # print(prediction_df.columns)
            # print(prediction_df.head())
            prediction_set = prediction_df.iloc[:, 1:]

            # interactime_train = drop class NA values
            training_set = interactome_edge_nodes.ix[
                pd.notnull(interactome_edge_nodes['class'])]

            # fit SVM on interactome_train
            model = svm.SVC(probability=True)
            model.fit(training_set['interactome_weight'].reshape(-1, 1),
                      training_set['class'])

            # predict on interactome_predict
            # predicted = model.predict(prediction_set)
            # print(predicted)

            # save prediction as confidence, not as class
            predicted_probab = model.predict_proba(prediction_set)
            # print(predicted_probab)

            # reformat name such that it is #tail head score
            predicted_probab_df = pd.DataFrame(predicted_probab)
            predicted_df = prediction_df['head'].\
                str.split('_to_', expand=True)
            predicted_df['pclass1'] = predicted_probab_df[1]
            predicted_df.columns = ['#tail', 'head', 'pclass1']
            predicted_df.sort_values('pclass1', ascending=False)
            # print(predicted_df['pclass1'].value_counts())
            # print(predicted_df.head())

            # write out prediction sorted by confidence value
            filename = '../output/{}-pathlearner_edgeranks_{}_of_{}.txt'.\
                format(pathway, fold_num, 2)
            predicted_df.to_csv(filename, sep='\t', index=False)
            print('{} created'.format(filename))
            # break  # break on folds loop

        # stack the fold values
        f1 = pd.read_csv('../output/{}-pathlearner_edgeranks_{}_of_{}.txt'.
                         format(pathway, 1, 2), delimiter='\t')
        f2 = pd.read_csv('../output/{}-pathlearner_edgeranks_{}_of_{}.txt'.
                         format(pathway, 2, 2), delimiter='\t')
        stacked = pd.concat([f1, f2], ignore_index=True)

        # sort on the score
        stacked = stacked.sort_values('pclass1', ascending=False)

        # save new file as path-edge-scores.txt
        fname = '../output/{}-pathlearner-edgeranks.txt'.format(pathway)
        stacked.to_csv(fname, index=False, sep='\t')
        print('{} created'.format(fname))

        # break  # break on pathways loop

if __name__ == '__main__':
    main()
