import CHP
import qubit_setup
import Noise
import preparation
import measurement_circuit
import sampling
import lattice
import fault_search
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
import csv

def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G,pos, with_labels=True)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show()

if __name__ == "__main__":
    distance = 3
    total_qubits = (2*distance-1)*(2*distance-1)
    QS = CHP.QuantumState(total_qubits)
    connections = qubit_setup.generate_connections(distance)
    x_ancilla_list = qubit_setup.generate_x_ancillas(distance, total_qubits)
    z_ancilla_list = qubit_setup.generate_z_ancillas(distance, total_qubits)
    data_qubit_list = qubit_setup.generate_data_qubits(total_qubits)
    measurement_circuit.apply_measurement_circuit(QS, distance, connections, x_ancilla_list, z_ancilla_list)
    stabilizers = QS.get_tableau()
    np.savetxt("stabilizers.csv", np.vstack(stabilizers), delimiter=",", fmt='%u')
    # with open("stabilizers.csv", "w", newline='') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(stabilizers)
    # initial_lattice = []
    # f = open("CSC_G_plaquette.txt", "r")
    # for x in f:
    #     initial_lattice.append(eval(x))
    # f.close()
    # final_lattice = lattice.error_lattice(distance, 4, initial_lattice)
    # G1 = fault_search.generate_shortest_path_graph_unique(distance, 4, final_lattice, [1,5])
    # G2 = fault_search.generate_shortest_path_graph(distance, 4, final_lattice, [1,5])
    # fault_search.update_weight(G1,100)
    # fault_search.update_weight(G2,100)
    # print(nx.max_weight_matching(G1, maxcardinality=True))
    # print(nx.max_weight_matching(G2, maxcardinality=True))