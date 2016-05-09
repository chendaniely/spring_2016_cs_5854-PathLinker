import pandas as pd
from sklearn import svm

from tqdm import tqdm

import helper

pathways, original_interactome = helper.setup()

kernels = ['linear', 'poly', 'rbf']

for kernel in tqdm(kernels):
    for pathway in tqdm(pathways):
        for fold_idx in tqdm(range(2)):
            fold_num = fold_idx + 1

            training = pd.read_csv(
                '../output/fit_training_{}_{}_of_2'.format(pathway, fold_num),
                delimiter='\t')
            testing = pd.read_csv(
                '../output/fit_prediction_{}_{}_of_2'.format(
                    pathway, fold_num),
                delimiter='\t')

            training = training.replace("None", 100000)
            testing = testing.replace("None", 100000)

            # fit SVM on interactome_train
            model = svm.SVC(probability=True, class_weight='balanced')
            model.fit(training.ix[:, 1:-1], training['class'])

            # save prediction as confidence, not as class
            predicted_probab = model.predict_proba(testing.ix[:, 1:-1])

            # reformat name such that it is #tail head score
            predicted_probab_df = pd.DataFrame(predicted_probab)
            predicted_df = testing['name'].\
                str.split('_to_', expand=True)
            predicted_df['pclass1'] = predicted_probab_df[1]
            predicted_df.columns = ['#tail', 'head', 'pclass1']
            predicted_df.sort_values('pclass1', ascending=False)

            # write out prediction sorted by confidence value
            filename = '../output/{}-pathlearner_edgeranks_{}_of_{}.txt'.\
                format(pathway, fold_num, 2)
            predicted_df.to_csv(filename, sep='\t', index=False)
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
        fname = '../output/99-{}-{}-pathlearner-edgeranks.txt'.format(
            kernel, pathway)
        stacked.to_csv(fname, index=False, sep='\t')
