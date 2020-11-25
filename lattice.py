import networkx as nx
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt 

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
