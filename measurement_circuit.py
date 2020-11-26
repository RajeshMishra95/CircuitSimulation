import CHP
# import Noise

def prep_x_ancillas(QS, x_ancilla_list):
    for i in x_ancilla_list:
        QS.apply_h(i)

def north_z_ancillas(QS, graph, z_ancilla_list):
    for j in z_ancilla_list:
        if j-1 in graph[j]:
            QS.apply_cnot(j-1, j)

def north_x_ancillas(QS, graph, x_ancilla_list):
    for j in x_ancilla_list:
        if j-1 in graph[j]:
            QS.apply_cnot(j, j-1)

def west_z_ancillas(QS, d, graph, z_ancilla_list):
    for j in z_ancilla_list:
        if j-(2*d-1) in graph[j]:
            QS.apply_cnot(j-(2*d-1), j)

def west_x_ancillas(QS, d, graph, x_ancilla_list):
    for j in x_ancilla_list:
        if j-(2*d-1) in graph[j]:
            QS.apply_cnot(j, j-(2*d-1))

def east_z_ancillas(QS, d, graph, z_ancilla_list):
    for j in z_ancilla_list:
        if j+(2*d-1) in graph[j]:
            QS.apply_cnot(j+(2*d-1), j)

def east_x_ancillas(QS, d, graph, x_ancilla_list):
    for j in x_ancilla_list:
        if j+(2*d-1) in graph[j]:
            QS.apply_cnot(j, j+(2*d-1))

def south_z_ancillas(QS, graph, z_ancilla_list):
    for j in z_ancilla_list:
        if j+1 in graph[j]:
            QS.apply_cnot(j+1, j)

def south_x_ancillas(QS, graph, x_ancilla_list):
    for j in x_ancilla_list:
        if j+1 in graph[j]:
            QS.apply_cnot(j, j+1)

def apply_measurement_circuit(QS, d, graph, x_ancilla_list, z_ancilla_list):
    # prepare the x_ancillas in the |+> state
    prep_x_ancillas(QS, x_ancilla_list)

    # carry out the measurement circuits
    # North
    north_z_ancillas(QS, graph, z_ancilla_list)

    north_x_ancillas(QS, graph, x_ancilla_list)

    # West
    west_z_ancillas(QS, d, graph, z_ancilla_list)

    west_x_ancillas(QS, d, graph, x_ancilla_list)

    # East
    east_z_ancillas(QS, d, graph, z_ancilla_list)

    east_x_ancillas(QS, d, graph, x_ancilla_list)

    # South
    south_z_ancillas(QS, graph, z_ancilla_list)

    south_x_ancillas(QS, graph, x_ancilla_list) 

    # measurement of the x_ancillas is the x basis
    for l in x_ancilla_list:
        QS.apply_h(l)

def ancilla_measurement(QS, x_ancilla_list, z_ancilla_list, measurement_value):
    for i in x_ancilla_list:
        measurement_value[i] = QS.measure(i)
    for i in z_ancilla_list:
        measurement_value[i] = QS.measure(i)
    return measurement_value
    