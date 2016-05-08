import pandas as pd

def convert_edges_to_node(data, weight_col='edge_weight'):
    """
    Takes a dataframe of edge lists and converts each edge into a node
    """
    data['edge_node'] = data['#tail'].astype(str) + \
                        '_to_' + \
                        data['head'].astype(str)

    new_edges_1 = data[['#tail', 'edge_node', weight_col]]
    new_edges_2 = data[['edge_node', 'head', weight_col]]

    new_edges_1.columns = ['#tail', 'head', 'edge_weight']
    new_edges_2.columns = ['#tail', 'head', 'edge_weight']

    new_data = pd.concat([new_edges_1, new_edges_2], ignore_index=True)
    return new_data
