import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(adj, pos, labels, nodelist=None, figsize=(12, 12), title=None):
    graph = nx.from_scipy_sparse_matrix(adj)
    plt.figure(figsize=figsize)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    nx.draw_networkx(graph, pos=pos, nodelist=nodelist, node_size=50, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=1, style='dotted', with_labels=False)
    plt.savefig("./images/{}.png".format(title))
    plt.show()
