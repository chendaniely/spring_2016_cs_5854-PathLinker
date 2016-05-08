import re

import pandas as pd
import numpy as np

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

#
# Begin Script
#

pathways, interactome = setup()

for pathway in pathways:
    part1 = pd.read_csv(
        '../output/features_{}_01.txt'.format(pathway), delimiter='\t')
    part2 = pd.read_csv(
        '../output/features_{}_02.txt'.format(pathway), delimiter='\t')
    part3 = pd.read_csv(
        '../output/features_{}_03.txt'.format(pathway), delimiter='\t')

    p1_en = part1.ix[part1['name'].str.match('.*_to_.*', as_indexer=True)]
    p2_en = part2.ix[part2['name'].str.match('.*_to_.*', as_indexer=True)]
    p3_en = part3.ix[part3['name'].str.match('.*_to_.*', as_indexer=True)]

    pathway_learning_df = pd.merge(p1_en, p2_en, on='name')
    pathway_learning_df = pd.merge(pathway_learning_df, p3_en, on='name')

    filename = '../output/features_{}'.format(pathway)
    pathway_learning_df.to_csv(filename, index=False, sep='\t')
    print('{} created.'.format(filename))
