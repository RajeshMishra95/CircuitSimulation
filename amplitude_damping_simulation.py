import numpy as np
import math
import csv
import random 
import matplotlib.pyplot as plt
import time
import copy

class QuantumState:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.tableau = np.concatenate((np.identity(2*num_qubits,dtype=int),np.zeros((2*num_qubits,1),dtype=int)),axis=1)
    
    def get_tableau(self):
        return self.tableau

    def apply_h(self, qubit):
        for i in range(2*self.num_qubits):
            self.tableau[i][2*self.num_qubits] = np.bitwise_xor(self.tableau[i][2*self.num_qubits], self.tableau[i][qubit]*self.tableau[i][qubit+self.num_qubits])
            self.tableau[i][qubit], self.tableau[i][qubit+self.num_qubits] = self.tableau[i][qubit+self.num_qubits], self.tableau[i][qubit]

    def apply_s(self, qubit):
        for i in range(2*self.num_qubits):
            self.tableau[i][2*self.num_qubits] = np.bitwise_xor(self.tableau[i][2*self.num_qubits], self.tableau[i][qubit]*self.tableau[i][qubit+self.num_qubits])
            self.tableau[i][qubit+self.num_qubits] = np.bitwise_xor(self.tableau[i][qubit+self.num_qubits],self.tableau[i][qubit])

    def apply_cnot(self, control_qubit, target_qubit):
        for i in range(2*self.num_qubits):
            self.tableau[i][2*self.num_qubits] = np.bitwise_xor(self.tableau[i][2*self.num_qubits],self.tableau[i][control_qubit]*self.tableau[i][target_qubit+self.num_qubits]*np.bitwise_xor(self.tableau[i][target_qubit],np.bitwise_xor(self.tableau[i][control_qubit+self.num_qubits],1)))
            self.tableau[i][target_qubit] = np.bitwise_xor(self.tableau[i][target_qubit],self.tableau[i][control_qubit])
            self.tableau[i][control_qubit+self.num_qubits] = np.bitwise_xor(self.tableau[i][control_qubit+self.num_qubits],self.tableau[i][target_qubit+self.num_qubits])
               
    def apply_z(self, qubit):
        self.apply_s(qubit)
        self.apply_s(qubit)

    def apply_x(self, qubit):
        self.apply_h(qubit)
        self.apply_z(qubit)
        self.apply_h(qubit)

    def apply_y(self, qubit):
        self.apply_z(qubit)
        self.apply_x(qubit)

    def apply_Rz(self, qubit):
        if self.measure(qubit) != [1,0]:
            self.apply_x(qubit)

    def __func_g(self,x1,z1,x2,z2):
        if (x1 == 0 and z1 == 0):
            return 0
        elif (x1 == 1 and z1 == 1):
            return z2 - x2
        elif (x1 == 1 and z1 == 0):
            return z2*(2*x2 - 1)
        else:
            return x2*(1 - 2*z2)

    def __rowsum(self,h,i):
        value = 2*self.tableau[h][2*self.num_qubits] + 2*self.tableau[i][2*self.num_qubits]
        for j in range(self.num_qubits):
            value += self.__func_g(self.tableau[i][j], self.tableau[i][j+self.num_qubits], self.tableau[h][j], self.tableau[h][j+self.num_qubits])

        if value%4 == 0:
            self.tableau[h][2*self.num_qubits] = 0
        elif value%4 == 2:
            self.tableau[h][2*self.num_qubits] = 1

    def prob_measure(self, qubit):

        is_random = False 
        p = 0
        for i in range(self.num_qubits, 2*self.num_qubits):
            if self.tableau[i][qubit] == 1:
                is_random = True
                p = i
                break

        if is_random:
            return [0.5,0.5,p]
        else:
            self.tableau = np.concatenate((self.tableau, np.zeros((1,2*self.num_qubits+1), dtype=int)))
            for i in range(self.num_qubits):
                if self.tableau[i][qubit] == 1:
                    self.__rowsum(2*self.num_qubits,i+self.num_qubits)
            if self.tableau[2*self.num_qubits][2*self.num_qubits] == 0:
                self.tableau = np.delete(self.tableau, (-1), axis=0)
                return [1,0]
            else:
                self.tableau = np.delete(self.tableau, (-1), axis=0)
                return [0,1]

    def measure(self,qubit):
        prob_measure = self.prob_measure(qubit)
        if prob_measure == [1,0]:
            return [1,0]
        elif prob_measure == [0,1]:
            return [0,1]
        else: 
            for i in range(2*self.num_qubits):
                if self.tableau[i][qubit] == 1 and (i != prob_measure[2]):
                    self.__rowsum(i,prob_measure[2])
            self.tableau[prob_measure[2]-self.num_qubits][:] = self.tableau[prob_measure[2]][:]
            self.tableau[prob_measure[2]][:] = np.zeros((1,2*self.num_qubits+1),dtype=int)
            self.tableau[prob_measure[2]][-1] = random.randint(0,1)
            self.tableau[prob_measure[2]][self.num_qubits+qubit] = 1
            if self.tableau[prob_measure[2]][-1] == 0:
                return [1,0]
            elif self.tableau[prob_measure[2]][-1] == 1:
                return [0,1]


def gateOperation(quantum_state, gate, qubit, num_qubits=1):
    if num_qubits == 1:
        if gate == 'H':
            quantum_state.apply_h(qubit[0])
        elif gate == 'S':
            quantum_state.apply_s(qubit[0])
        elif gate == 'X':
            quantum_state.apply_x(qubit[0])
        elif gate == 'Y':
            quantum_state.apply_y(qubit[0])
        elif gate == 'Z':
            quantum_state.apply_z(qubit[0])
        elif gate == 'Rz':
            quantum_state.apply_Rz(qubit[0])
    elif num_qubits == 2:
        quantum_state.apply_cnot(qubit[0], qubit[1])

def stabgateOperation(quantum_state, coeff_gates, prob_gates, gates, qubit, w):
    i = samplingDistribution(prob_gates)
    w *= coeff_gates[i]/prob_gates[i]
    gateOperation(quantum_state, gates[i], [qubit], num_qubits=1)
    return w

def observableEstimation(quantum_state, coeff_gates, gates, num_runs, num_qubits):
       
    prob_gates = probDistribution(coeff_gates)

    expectation_value = [0 for i in range(num_qubits)]

    for j in range(num_runs):
        
        w = [1 for i in range(num_qubits)]

        rho_star = copy.deepcopy(quantum_state)
#        for i in range(num_qubits):
#            gateOperation(rho_star,'H',[i],num_qubits=1)
#        for i in range(num_qubits):
#            gateOperation(rho_star,'H',[i],num_qubits=1)
        for i in range(num_qubits):
            w[i] = stabgateOperation(rho_star,coeff_gates,prob_gates,gates,i,w[i])
#        gateOperation(rho_star,'CNOT',[1,0],num_qubits=2)
        for i in range(num_qubits):
            w[i] = stabgateOperation(rho_star,coeff_gates,prob_gates,gates,i,w[i])
#        gateOperation(rho_star,'CNOT',[1,0],num_qubits=2)
#        for i in range(num_qubits):
#            w[i] = stabgateOperation(rho_star,coeff_gates,prob_gates,gates,i,w[i])
#        gateOperation(rho_star,'H',[1],num_qubits=1)
#        for i in range(num_qubits):
#            w[i] = stabgateOperation(rho_star,coeff_gates,prob_gates,gates,i,w[i])

        prob_outcome = [0 for i in range(num_qubits)]

        for i in range(num_qubits):
            prob_outcome[i] = rho_star.prob_measure(i)
            expectation_value[i] += w[i]*prob_outcome[i][0]/num_runs
    
    return expectation_value

        
def oneNorm(vector):
    '''Calculates the 1-norm of the given vector'''
    one_norm_vector = float(0)
    for i in vector:
        one_norm_vector += abs(i)
    return one_norm_vector

def probDistribution(quasi_prob):
    '''converts the quasiprobability distrbution to a probability distribution to allow for sampling'''
    prob = []
    for i in quasi_prob:
        prob.append(abs(i)/oneNorm(quasi_prob))
    return prob

def samplingDistribution(pdf):
    '''sampling a value from the given distribution.'''
    cdf = [0]
    for i in range(len(pdf)):
        cdf.append(cdf[i]+pdf[i])
    
    random_value = random.random()
    idx = 0

    for i in range(len(cdf)):
        if random_value <= cdf[i]:            
            idx = i-1
            break
    return idx

   

if __name__ == "__main__":

    time_taken = []
    for i in range(1,101):
        num_qubits = i

#        with open('gate_set.csv') as csv_file:
#            gate_set = list(csv.reader(csv_file, delimiter=','))
#    for i in range(len(f)):
#        f[i] = f[i][0].split("_")
#    with open('measurement_set.csv') as csv_file:
#        f = list(csv.reader(csv_file))
#    for i in range(len(f)):
#        f[i] = f[i][0].split("_")
        
        
        QS = QuantumState(num_qubits)

#        gamma = 0.05
#        coeff_gates = [(1-gamma)/2+np.sqrt(1-gamma)/2, (1-gamma)/2-np.sqrt(1-gamma)/2, gamma]
        for j in range(i):
            gateOperation(QS,'H',[j],1)

        start = time.time()
        for j in range(1000):
            QS.measure(i)
#    for i in range(100):
#        start = time.time()
#
#    for i in range(10):
#        observableEstimation(QS, coeff_gates, gate_set[0], 100000, num_qubits)
#        value.append(exp_list[-1])
#        for j in range(num_qubits):
#            QS.measure(j)
        end = time.time()
        time_taken.append(end-start)
        print(i)

    with open('time_taken_gates1.csv', 'w',newline='') as filehandle:
        wr = csv.writer(filehandle)
        for item in time_taken:
            wr.writerow([item])
