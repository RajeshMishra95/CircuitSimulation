import networkx as nx
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt 
import itertools

def error_lattice(dist, cycles, initial_lattice):
    """
    Builds the volume lattice using the initial lattice
    """
    if __debug__:
        if not (cycles > 2): raise AssertionError 
    final_lattice = deepcopy(initial_lattice)
    inc = dist*(dist+1)
    for j in range(1, cycles-1):
        for i in initial_lattice:
            final_lattice.append((i[0]+j*inc, i[1]+j*inc))
    return final_lattice

def build_lattice_graph(d, cycles, edges):
    G = nx.Graph()
    total_nodes = d*(d+1)*cycles
    for i in range(1,total_nodes+1):
        G.add_node(i)
    G.add_edges_from(edges)
    return G

# def generate_fault_nodes(num_cycles, measurement_cycles):
#     """
#     Generates a list of nodes (vertex and plaquettes) with faults 
#     numbered in the form of the volume lattice. Measurement cycles
#     are measurement values for multiple cycles. Values for each 
#     cycle are stored in an array numbered from 0 to 2d*(d+1).
#     Actual numbering is from 1 to 2d*(d+1) + 1.
#     """


def generate_shortest_path_graph_vertex(d, cycles, volume_lattice, fault_nodes_x):
    """
    Takes the fault node and generates a graph containing the fault nodes and the 
    shortest distance between each of the fault nodes and also bwteen the fault nodes
    their corresponding spatial and temporal ghost nodes.
    """
    Graph_volume_lattice = build_lattice_graph(d, cycles, volume_lattice)
    Graph_vertex = nx.Graph()
    total_nodes = 3*len(fault_nodes_x)
    for i in fault_nodes_x:
        Graph_vertex.add_node(i)
    for pair in itertools.combinations(fault_nodes_x, 2):
        w = nx.shortest_path_length(Graph_volume_lattice, source=pair[0],
        target=pair[1])
        Graph_vertex.add_edge(*pair, weight=w)

    # Adding the spatial ghost nodes
    total_nodes_per_layer = d*(d+1)
    total_real_nodes = d*(d-1)
    for i in fault_nodes_x:
        spatial_ghost_nodes = range((int(i/total_nodes_per_layer))*total_nodes_per_layer + total_real_nodes + 1,
        (int(i/total_nodes_per_layer) + 1)*total_nodes_per_layer + 1)
        all_shortest_paths = []
        for j in spatial_ghost_nodes:
            all_shortest_paths.append(nx.shortest_path_length(Graph_volume_lattice,
            source=i, target=j))
        Graph_vertex.add_node(spatial_ghost_nodes[min(range(len(all_shortest_paths)),
        key=all_shortest_paths.__getitem__)])
        Graph_vertex.add_edge(i,spatial_ghost_nodes[min(range(len(all_shortest_paths)),
        key=all_shortest_paths.__getitem__)], weight=min(all_shortest_paths))
        # Adding temporal ghost nodes
        t_node = i + (cycles - 1 - int(i/total_nodes_per_layer))*total_nodes_per_layer
        Graph_vertex.add_node(t_node)
        Graph_vertex.add_edge(i, t_node, weight=nx.shortest_path_length(Graph_volume_lattice, source=i,
        target=t_node))

    return Graph_vertex

def generate_shortest_path_graph_plaquette(d, cycles, volume_lattice, fault_nodes_z):
    """
    Takes the fault node and generates a graph containing the fault nodes and the 
    shortest distance between each of the fault nodes and also bwteen the fault nodes
    their corresponding spatial and temporal ghost nodes.
    """
    Graph_volume_lattice = build_lattice_graph(d, cycles, volume_lattice)
    Graph_vertex = nx.Graph()
    total_nodes = 3*len(fault_nodes_z)
    for i in fault_nodes_z:
        Graph_vertex.add_node(i)
    for pair in itertools.combinations(fault_nodes_z, 2):
        w = nx.shortest_path_length(Graph_volume_lattice, source=pair[0],
        target=pair[1])
        Graph_vertex.add_edge(*pair, weight=w)

    # Adding the spatial ghost nodes
    total_nodes_per_layer = d*(d+1)
    total_real_nodes = d*(d-1)
    for i in fault_nodes_z:
        spatial_ghost_nodes = range((int(i/total_nodes_per_layer))*total_nodes_per_layer + total_real_nodes + 1,
        (int(i/total_nodes_per_layer) + 1)*total_nodes_per_layer + 1)
        all_shortest_paths = []
        for j in spatial_ghost_nodes:
            all_shortest_paths.append(nx.shortest_path_length(Graph_volume_lattice,
            source=i, target=j))
        Graph_vertex.add_node(spatial_ghost_nodes[min(range(len(all_shortest_paths)),
        key=all_shortest_paths.__getitem__)])
        Graph_vertex.add_edge(i,spatial_ghost_nodes[min(range(len(all_shortest_paths)),
        key=all_shortest_paths.__getitem__)], weight=min(all_shortest_paths))
        # Adding temporal ghost nodes
        t_node = i + (cycles - 1 - int(i/total_nodes_per_layer))*total_nodes_per_layer
        Graph_vertex.add_node(t_node)
        Graph_vertex.add_edge(i, t_node, weight=nx.shortest_path_length(Graph_volume_lattice, source=i,
        target=t_node))
    
    return Graph_vertex

if __name__ == "__main__":
    distance = 3
    initial_lattice = []
    f = open("CSC_G_plaquette1.txt", "r")
    for x in f:
        initial_lattice.append(eval(x))
    f.close()
    final_lattice = error_lattice(distance, 3, initial_lattice)
    G = generate_shortest_path_graph_plaquette(distance, 3, final_lattice, [1,5,13,17])
    pos = nx.spring_layout(G)
    nx.draw(G,pos,with_labels=True)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show()    