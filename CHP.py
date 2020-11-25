import numpy as np
import math
import random

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

    def __prob_measure(self, qubit):

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
        prob_measure = self.__prob_measure(qubit)
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