def generate_connections(distance):
    d = distance
    dim = 2*d - 1
    total_qubits = dim*dim
    connections_dict = {}
    for i in range(1, total_qubits + 1):
        list_connections = []
        if i%dim == 0:
            if i-dim > 0:
                list_connections.append(i-dim)
            list_connections.append(i-1)
            if i+dim < total_qubits + 1:
                list_connections.append(i+dim)
        elif i%dim == 1:
            if i-dim > 0:
                list_connections.append(i-dim)
            list_connections.append(i+1)
            if i+dim < total_qubits + 1:
                list_connections.append(i+dim)
        else:
            if i-dim > 0:
                list_connections.append(i-dim)
            list_connections.append(i-1)
            list_connections.append(i+1)
            if i+dim < total_qubits+1:
                list_connections.append(i+dim)
        connections_dict[i] = list_connections
    return connections_dict

def generate_data_qubits(total_qubits):
    data_qubits = []
    for i in range(1, total_qubits+1):
        if i%2 != 0:
            data_qubits.append(i)
    return data_qubits

def generate_x_ancillas(distance, total_qubits):
    x_ancillas = []
    for i in range(1,distance):
        x_ancillas.append(2*i)
    for j in x_ancillas:
        if j + 4*distance - 2 < total_qubits:
            x_ancillas.append(j + 4*distance -2)
    return x_ancillas

def generate_z_ancillas(distance, total_qubits):
    z_ancillas = [2*distance]
    for i in range(1,distance):
        z_ancillas.append(z_ancillas[-1] + 2)
    for j in z_ancillas:
        if j + 4*distance - 2 < total_qubits:
            z_ancillas.append(j + 4*distance -2)
    return z_ancillas
