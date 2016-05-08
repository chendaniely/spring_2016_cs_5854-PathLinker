import pandas as pd

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
    pathway_df = pd.read_csv('../data/pathways/{}-edges.txt'.
                             format(pathway),
                             delimiter='\t')
    pagerank_df = pd.read_csv('../data/pagerank/{}-q_0.50-edge-fluxes.txt'.
                              format(pathway),
                              delimiter='\t')
    cyclinker_df = pd.read_csv('../data/cyclinker/{}-k_110000-ranked-edges.txt'.
                               format(pathway),
                               delimiter='\t')

    pathway_df_e2n = helper.convert_edges_to_node(
        pathway_df, weight_col_name='pathway_value')
    pagerank_df_e2n = helper.convert_edges_to_node(
        pagerank_df, 'edge_flux', 'pagerank_value')
    cyclinker_df_e2n = helper.convert_edges_to_node(
        cyclinker_df, 'KSP index', 'cyclinker_value')

    pathway_en = helper.keep_edge_nodes(
        pathway_df_e2n, ['head', 'pathway_value'])
    pagerank_en = helper.keep_edge_nodes(
        pagerank_df_e2n, ['head', 'pagerank_value'])
    cyclinker_en = helper.keep_edge_nodes(
        cyclinker_df_e2n, ['head', 'cyclinker_value'])

    pathway_ranks = pd.merge(
        pathway_en, pagerank_en, on='head', how='left')
    pathway_ranks = pd.merge(
        pathway_ranks, cyclinker_en, on='head', how='left')

    pathway_ranks.ix[pd.isnull(pathway_ranks['pagerank_value']),
                     'pagerank_value'] = pathway_ranks['pagerank_value'].\
        min()

    pathway_ranks.ix[pd.isnull(pathway_ranks['cyclinker_value']),
                     'cyclinker_value'] = pathway_ranks['cyclinker_value'].\
        max() + 1

    pathway_ranks.columns = ['name', 'pathway_value',
                             'pagerank_value', 'cyclinker_value']

    filename = '../output/features_{}_03.txt'.format(pathway)
    pathway_ranks.to_csv(filename, index=False, sep='\t')
    print('{} created'.format(filename))
