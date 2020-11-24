import networkx as nx
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt 
import itertools

def error_lattice(dist, cycles, initial_lattice):
    """
    Creats the adjacency list of the volume lattice using the initial lattice.
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
    """ Generates a graph given the arguments."""
    G = nx.Graph()
    total_nodes = d*(d+1)*cycles
    for i in range(1,total_nodes+1):
        G.add_node(i)
    G.add_edges_from(edges)
    return G

def generate_shortest_path_graph(d, cycles, volume_lattice, fault_nodes):
    """
    Takes the fault node and generates a graph containing the fault nodes and the 
    shortest distance between each of the fault nodes and also between the fault nodes
    their corresponding spatial and temporal ghost nodes. Here, ghost can be shared by 
    multiple real nodes.
    """
    Graph_volume_lattice = build_lattice_graph(d, cycles, volume_lattice)
    Graph_fault = nx.Graph()
    for i in fault_nodes:
        Graph_fault.add_node(i)
    for pair in itertools.combinations(fault_nodes, 2):
        w = nx.shortest_path_length(Graph_volume_lattice, source=pair[0],
        target=pair[1])
        Graph_fault.add_edge(*pair, weight=w)

    # Adding the spatial ghost nodes
    total_nodes_per_layer = d*(d+1)
    total_real_nodes = d*(d-1)
    all_ghost_nodes = []
    for i in fault_nodes:
        spatial_ghost_nodes = range((int(i/total_nodes_per_layer))*total_nodes_per_layer + total_real_nodes + 1,
        (int(i/total_nodes_per_layer) + 1)*total_nodes_per_layer + 1)
        all_shortest_paths = []
        for j in spatial_ghost_nodes:
            all_shortest_paths.append(nx.shortest_path_length(Graph_volume_lattice,
            source=i, target=j))
        all_ghost_nodes.append(spatial_ghost_nodes[min(range(len(all_shortest_paths)),
        key=all_shortest_paths.__getitem__)])
        Graph_fault.add_node(spatial_ghost_nodes[min(range(len(all_shortest_paths)),
        key=all_shortest_paths.__getitem__)])
        Graph_fault.add_edge(i,spatial_ghost_nodes[min(range(len(all_shortest_paths)),
        key=all_shortest_paths.__getitem__)], weight=min(all_shortest_paths))
        # Adding temporal ghost nodes
        t_node = i + (cycles - 1 - int(i/total_nodes_per_layer))*total_nodes_per_layer
        all_ghost_nodes.append(t_node)
        Graph_fault.add_node(t_node)
        Graph_fault.add_edge(i, t_node, weight=nx.shortest_path_length(Graph_volume_lattice, source=i,
        target=t_node))

    for pair in itertools.combinations(all_ghost_nodes, 2):
        Graph_fault.add_edge(*pair, weight=0)

    if len(Graph_fault)%2 == 1:
        Graph_fault.add_node('D', value=0)
        for i in all_ghost_nodes:
            Graph_fault.add_edge('D', i, weight=0)

    return Graph_fault

def generate_shortest_path_graph_unique(d, cycles, volume_lattice, fault_nodes):
    """
    Takes the fault node and generates a graph containing the fault nodes and the 
    shortest distance between each of the fault nodes and also between the fault nodes
    their corresponding spatial and temporal ghost nodes. Here, ghost nodes are not 
    shared by multiple real nodes.
    """
    Graph_volume_lattice = build_lattice_graph(d, cycles, volume_lattice)
    Graph_fault = nx.Graph()
    for i in fault_nodes:
        Graph_fault.add_node(str(i), value=i)
    for pair in itertools.combinations(fault_nodes, 2):
        w = nx.shortest_path_length(Graph_volume_lattice, source=pair[0],
        target=pair[1])
        Graph_fault.add_edge(str(pair[0]), str(pair[1]), weight=w)

    # Adding the spatial ghost nodes
    total_nodes_per_layer = d*(d+1)
    total_real_nodes = d*(d-1)
    all_ghost_nodes = []
    for i in fault_nodes:
        spatial_ghost_nodes = range((int(i/total_nodes_per_layer))*total_nodes_per_layer + total_real_nodes + 1,
        (int(i/total_nodes_per_layer) + 1)*total_nodes_per_layer + 1)
        all_shortest_paths = []
        for j in spatial_ghost_nodes:
            all_shortest_paths.append(nx.shortest_path_length(Graph_volume_lattice,
            source=i, target=j))
        all_ghost_nodes.append('S'+str(i))
        Graph_fault.add_node(all_ghost_nodes[-1], value=spatial_ghost_nodes[min(range(len(all_shortest_paths)),
        key=all_shortest_paths.__getitem__)])
        Graph_fault.add_edge(str(i),all_ghost_nodes[-1], weight=min(all_shortest_paths))
        # Adding temporal ghost nodes
        t_node = i + (cycles - 1 - int(i/total_nodes_per_layer))*total_nodes_per_layer
        all_ghost_nodes.append('T'+str(i))
        Graph_fault.add_node(all_ghost_nodes[-1], value=t_node)
        Graph_fault.add_edge(str(i),all_ghost_nodes[-1], weight=nx.shortest_path_length(Graph_volume_lattice, source=i,
        target=t_node))
    
    for pair in itertools.combinations(all_ghost_nodes, 2):
        Graph_fault.add_edge(*pair, weight=0)

    if len(Graph_fault)%2 == 1:
        Graph_fault.add_node('D', value=0)
        for i in all_ghost_nodes:
            Graph_fault.add_edge('D', i, weight=0)
    
    return Graph_fault

def update_weight(graph, value):
    """
    This is done so that we can find the minimum weight matching by using the
    maximum weight matching algorithm.
    """
    for (u,v) in graph.edges():
        graph[u][v]['weight'] = value - graph[u][v]['weight'] 


if __name__ == "__main__":
    distance = 3
    initial_lattice = []
    f = open("CSC_G_plaquette.txt", "r")
    for x in f:
        initial_lattice.append(eval(x))
    f.close()
    final_lattice = error_lattice(distance, 4, initial_lattice)
    G1 = generate_shortest_path_graph_unique(distance, 4, final_lattice, [1,5])
    G2 = generate_shortest_path_graph(distance, 4, final_lattice, [1,5])
    update_weight(G1,100)
    update_weight(G2,100)
    print(nx.max_weight_matching(G1, maxcardinality=True))
    print(nx.max_weight_matching(G2, maxcardinality=True))
    # pos = nx.spring_layout(G2)
    # nx.draw(G2,pos, with_labels=True)
    # labels = nx.get_edge_attributes(G2,'weight')
    # nx.draw_networkx_edge_labels(G2,pos,edge_labels=labels)
    # plt.show()    
