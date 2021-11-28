import dwave.inspector
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.code.quantum_annealing.results.result import Result
from src.code.model.graph import TSP_Graph


def print_tsp_graph(tsp_graph: TSP_Graph):
    G = tsp_graph.G
    edges = G.edges
    colors = [G[u][v]['color'] for u, v in edges]
    widths = [G[u][v]['width'] for u, v in edges]
    nx.draw(G, labels=nx.get_node_attributes(G, 'name'), with_labels=True, edge_color=colors, width=widths)
    plt.show()


def print_tsp_solutions(tsp_graph, solutions):
    # todo: it should be possible to add a legend specifying the name of each node numbered as an int
    G = tsp_graph.G
    sol_graph = nx.Graph()
    sol_graph.add_nodes_from(list(G.nodes))

    for sol in solutions:
        sol_graph.remove_edges_from(sol_graph.edges())
        color_map = ['z'] * len(sol_graph.nodes)
        lines = []

        last = 0
        for i in range(len(sol) - 1):
            if i == 0:
                color_map[(sol[i])[0]] = 'red'
            else:
                color_map[(sol[i])[0]] = 'green'

            if (sol[i])[0] >= (sol[i + 1])[0]:
                low = (sol[i + 1])[0]
                high = (sol[i])[0]
            else:
                low = (sol[i])[0]
                high = (sol[i + 1])[0]
            distance = G.get_edge_data(low, high)['distance']
            sol_graph.add_edge(low, high, distance=distance)
            last = i

        # lastly we color the ending node as blue and draw the network
        color_map[(sol[last + 1])[0]] = 'blue'
        lines.append(last + 1)

        pos = nx.spring_layout(sol_graph)
        nx.draw(sol_graph, pos, node_color=color_map, with_labels=True)
        node_labels = nx.get_node_attributes(sol_graph, 'name')
        nx.draw_networkx_labels(sol_graph, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(sol_graph, 'distance')
        nx.draw_networkx_edge_labels(sol_graph, pos, edge_labels=edge_labels)
        plt.show()


def embedding_inspector(qpu_res: Result):
    # if we have a decorator incompatible version for dwave-networkx just upgrade it
    # pip install dwave-networkx --upgrade
    dwave.inspector.show(qpu_res.response)


def histogram_energies(sampleset_SA, sampleset_QPU):
    # Plot energy histograms for both QPUs
    num_bins = 100
    use_bin = 50

    fig = plt.figure(figsize=(8, 5))
    SA = sampleset_SA.record.energy
    QPU = sampleset_QPU.record.energy

    bins=np.histogram(np.hstack((SA, QPU)), bins=num_bins)[1]

    ax = fig.add_subplot(1, 1, 1)

    ax.hist(SA, bins[0:use_bin], color='g', alpha=0.4, label="SA")
    ax.hist(QPU, bins[0:use_bin], color='r', alpha=0.4, label="QPU")

    ax.set_xlabel("Energy")
    ax.set_ylabel("Samples")
    ax.legend()
    plt.show()
