import CHP
import sampling
import lattice
import fault_search
import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G,pos, with_labels=True)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show()

if __name__ == "__main__":
    distance = 3
    initial_lattice = []
    f = open("CSC_G_plaquette.txt", "r")
    for x in f:
        initial_lattice.append(eval(x))
    f.close()
    final_lattice = lattice.error_lattice(distance, 4, initial_lattice)
    G1 = fault_search.generate_shortest_path_graph_unique(distance, 4, final_lattice, [1,5])
    G2 = fault_search.generate_shortest_path_graph(distance, 4, final_lattice, [1,5])
    fault_search.update_weight(G1,100)
    fault_search.update_weight(G2,100)
    print(nx.max_weight_matching(G1, maxcardinality=True))
    print(nx.max_weight_matching(G2, maxcardinality=True))