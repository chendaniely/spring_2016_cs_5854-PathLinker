import pandas as pd

from tqdm import tqdm

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

for pathway in tqdm(pathways):
    pathway_dist_en = pd.read_csv('../output/features_{}_02.txt'.
                                  format(pathway),
                                  delimiter='\t')

    pagerank_df = pd.read_csv('../data/pagerank/{}-q_0.50-edge-fluxes.txt'.
                              format(pathway),
                              delimiter='\t')
    cyclinker_df = pd.read_csv('../data/cyclinker/{}-k_110000-ranked-edges.txt'.
                               format(pathway),
                               delimiter='\t')

    pagerank_df_e2n = helper.convert_edges_to_node(
        pagerank_df, 'edge_flux', 'pagerank_value')
    cyclinker_df_e2n = helper.convert_edges_to_node(
        cyclinker_df, 'KSP index', 'cyclinker_value')

    pagerank_en = helper.keep_edge_nodes(
        pagerank_df_e2n, ['head', 'pagerank_value'])
    cyclinker_en = helper.keep_edge_nodes(
        cyclinker_df_e2n, ['head', 'cyclinker_value'])

    pathway_dist_ranks = pd.merge(
        pathway_dist_en, pagerank_en,
        left_on='name', right_on='head', how='left')

    pathway_dist_ranks = pd.merge(
        pathway_dist_ranks, cyclinker_en,
        left_on='name', right_on='head', how='left')

    pathway_dist_ranks.ix[
        pd.isnull(pathway_dist_ranks['pagerank_value']),
        'pagerank_value'] = pathway_dist_ranks['pagerank_value'].min()

    pathway_dist_ranks.ix[
        pd.isnull(pathway_dist_ranks['cyclinker_value']),
        'cyclinker_value'] = pathway_dist_ranks['cyclinker_value'].max() * 10

    pathway_dist_ranks.drop(['head_x', 'head_y'], axis=1, inplace=True)

    filename = '../output/features_{}_03.txt'.format(pathway)
    pathway_dist_ranks.to_csv(filename, index=False, sep='\t')
    print('{} created'.format(filename))
