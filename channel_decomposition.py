'''
Decomposition of an arbitrary channel into the standard basis containing 16 basis 
operations. The expression for the decomposition is a linear sum of all the basis
operations. The aim is to minimise the one-norm of all the coefficients. This 
leads to a minimisation of the negativity. This problem is transformed into a
linear optimisation problem.
'''

import numpy as np 
import scipy 

def pauli_transfer_matrix_gate(gate):
    '''
    This function generates the pauli transfer matrix for a 1-qubit channel.
    Gate: 2x2 matrix
    Output: 4x4 matrix
    '''
    iden = np.identity(2)
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, 0-1j], [0+1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    P = [iden, pauli_x, pauli_y, pauli_z]
    ptm = np.ones((4,4))
    for i in range(4):
        for j in range(4):
            ptm[i,j] = np.trace(P[i]*gate*P[j])
    
    return ptm

def pauli_transfer_matrix_channel(channel):
    '''
    This function generates the pauli transfer matrix for a 1-qubit channel.
    Channel: List of Kraus operators (2x2 matrix)
    Output: 4x4 matrix
    '''
    iden = np.identity(2)
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, 0-1j], [0+1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    P = [iden, pauli_x, pauli_y, pauli_z]
    kraus_operators = kraus_operation(channel)
    ptm = np.ones((4,4), dtype='complex128')
    for i in range(4):
        for j in range(4):
            ptm[i,j] = np.trace(P[i]*kraus_operators[j])
    
    return ptm

def kraus_operation(kraus_operators):
    iden = np.identity(2)
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, 0-1j], [0+1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    P = [iden, pauli_x, pauli_y, pauli_z]
    operators = []
    for i in range(4):
        operator = np.ones((2,2), dtype='complex128')
        for m in kraus_operators:
            operator += m*P[i]*np.conj(m)
        operators.append(operator)
    
    return operators

def basis_set():
    iden = np.identity(2)
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, 0-1j], [0+1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    R_x = np.sqrt(2)*(iden + 1j*pauli_x)
    R_y = np.sqrt(2)*(iden + 1j*pauli_y)
    R_z = np.sqrt(2)*(iden + 1j*pauli_z)
    R_xy = np.sqrt(2)*(pauli_x + pauli_y)
    R_yz = np.sqrt(2)*(pauli_y + pauli_z)
    R_zx = np.sqrt(2)*(pauli_z + pauli_x)
    pi_x = 0.5*(iden + pauli_x)
    pi_y = 0.5*(iden + pauli_y)
    pi_z = 0.5*(iden + pauli_z)
    pi_xy = 0.5*(pauli_x + 1j*pauli_y)
    pi_yz = 0.5*(pauli_y + 1j*pauli_z)
    pi_zx = 0.5*(pauli_z + 1j*pauli_x)
    basis = [iden, pauli_x, pauli_y, pauli_z, R_x, R_y, R_z, R_xy, R_yz, R_zx,
    pi_x, pi_y, pi_z, pi_xy, pi_yz, pi_zx]
    ptm_basis = []
    for i in basis:
        ptm_basis.append(pauli_transfer_matrix(i))
    return ptm_basis


# def channel_decomposition(channel):
#     ptm_basis = basis_set()
#     coeff = np.zeros(len(ptm_basis))
    
def amplitude_damping(gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    ad_channel = pauli_transfer_matrix_channel([K0, K1])
    c = 16*[1] + 16*[0]
    
    return ad_channel

# print(pauli_transfer_matrix(np.array([[0, 1], [1, 0]])))