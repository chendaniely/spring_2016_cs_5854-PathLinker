import pandas as pd
from sklearn import svm

from tqdm import tqdm

import helper

pathways, original_interactome = helper.setup()

prepend = 'set01_ori_balanced'
kernels = ['rbf']  # , 'linear', 'poly']

drop_cols = []
class_weight = 'balanced'  # 'balanced' or None

print('prepend: {}\nkernels: {}\ndrop_cols: {}\nclass_weight:{}'.format(
    prepend, kernels, drop_cols, class_weight))

for kernel in tqdm(kernels):
    for pathway in tqdm(pathways):
        for fold_idx in tqdm(range(2)):
            fold_num = fold_idx + 1

            training = pd.read_csv(
                '../output/fit_training_{}_{}_of_2'.format(
                    pathway, fold_num),
                delimiter='\t')
            testing = pd.read_csv(
                '../output/fit_prediction_{}_{}_of_2'.format(
                    pathway, fold_num),
                delimiter='\t')

            if drop_cols:
                ncols = training.shape[1]
                training.drop(drop_cols, axis=1, inplace=True)
                assert training.columns.shape[0] == ncols - len(drop_cols)

                ncols = testing.shape[1]
                testing.drop(drop_cols, axis=1, inplace=True)
                assert testing.columns.shape[0] == ncols - len(drop_cols)

            # fit SVM on interactome_train
            model = svm.SVC(kernel=kernel,
                            probability=True,
                            class_weight=class_weight)

            train_on = training.ix[
                :, ~training.columns.isin(['name', 'class'])]

            test_on = testing.ix[
                :, ~testing.columns.isin(['name', 'class'])]

            model.fit(train_on, training['class'])
            # save prediction as confidence, not as class
            predicted_probab = model.predict_proba(test_on)

            # reformat name such that it is #tail head score
            predicted_probab_df = pd.DataFrame(predicted_probab)
            predicted_df = testing['name'].\
                str.split('_to_', expand=True)
            predicted_df['pclass1'] = predicted_probab_df[1]
            predicted_df.columns = ['#tail', 'head', 'pclass1']
            predicted_df.sort_values('pclass1', ascending=False)

            # write out prediction sorted by confidence value
            filename = '../output/{}-{}-pathlearner_edgeranks_{}_of_{}.txt'.\
                format(prepend, pathway, fold_num, 2)
            predicted_df.to_csv(filename, sep='\t', index=False)
            # break  # break on folds loop

        # stack the fold values
        f1 = pd.read_csv('../output/{}-{}-pathlearner_edgeranks_{}_of_{}.txt'.
                         format(prepend, pathway, 1, 2), delimiter='\t')
        f2 = pd.read_csv('../output/{}-{}-pathlearner_edgeranks_{}_of_{}.txt'.
                         format(prepend, pathway, 2, 2), delimiter='\t')
        stacked = pd.concat([f1, f2], ignore_index=True)

        # sort on the score
        stacked = stacked.sort_values('pclass1', ascending=False)

        # save new file as path-edge-scores.txt
        fname = '../output/99-{}-{}-{}-pathlearner-edgeranks_full.txt'.\
                format(prepend, kernel, pathway)
        stacked.to_csv(fname, index=False, sep='\t')
