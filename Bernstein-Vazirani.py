#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pyquil import get_qc, Program
from pyquil.gates import CNOT, Z, X, H, I, MEASURE
from pyquil.api import local_forest_runtime, QuantumComputer
from pyquil.quil import DefGate

import numpy as np
from collections import defaultdict
from operator import xor
from typing import Dict, Tuple
import pprint

import argparse, sys

parser=argparse.ArgumentParser()
parser.add_argument('--a', help='Enter the a bit string for the function ax(xor)b you want to run with')
parser.add_argument('--b', help='Enter the b sibgle bit for the function ax(xor)b you want to run with')


# In[3]:


def create_bv_bitmap(a, b):
    n_bits = len(a)
    bit_map = {}
    for bit_val in range(2 ** n_bits):
        bit_map[np.binary_repr(bit_val, width=n_bits)] = str(
            (sum([int(np.binary_repr(bit_val, width=n_bits)[i]) * int(a[i]) for i in range(n_bits)]) % 2
            + int(b, 2)) % 2
        )

    return bit_map


class Bernstein_Vazirani(object):
    def __init__(self):
        self.n_qubits = None
        self.n_helper = 1
        self.qubits = None
        self.helper = None
        self.n_trials = None

    def _create_oracle(self, bit):
        oracle = np.zeros(shape=(2 ** (self.n_qubits + 1), 2 ** (self.n_qubits + 1)))

        for b in range(2**self.n_helper):
            bin_str = np.binary_repr(b, width=self.n_qubits)
            for k, v in bit.items():
                i, j = int(bin_str+k, 2), int(np.binary_repr(xor(int(bin_str, 2),
                                                                 int(v, 2)), self.n_qubits) + k, 2)
                oracle[i, j] = 1
        return oracle
    
    def _run_init(self, bit):
        self.n_qubits = len(list(bit.keys())[0])
        self.qubits = list(range(self.n_qubits))
        self.helper = self.n_qubits
        self.n_trials = 10

    def run_bv(self, bit):
        # initialize all attributes
        self._run_init(bit)
        qvm = get_qc('9q-square-qvm')
        
        # To get a
        oracle = self._create_oracle(bit)
        a_circuit = Program()
        a_ro = a_circuit.declare('ro', 'BIT', len(self.qubits) + 1)
        
        bv_circuit = Program()
        bv_circuit.defgate("oracle", oracle)
        bv_circuit.inst(X(self.helper), H(self.helper))
        bv_circuit.inst([H(i) for i in self.qubits])
        bv_circuit.inst(tuple(["oracle"] + sorted(self.qubits + [self.helper], reverse=True)))
        bv_circuit.inst([H(i) for i in self.qubits])
        
        a_circuit += bv_circuit
        a_circuit += [MEASURE(qubit, ro) for qubit, ro in zip(self.qubits,a_ro)]
        
        a_executable = qvm.compile(a_circuit)
        
        for i in range(self.n_trials):
            a_results = qvm.run(a_executable)
            print("trial {} a:".format(i))
            pprint.pprint(a_results[0][::-1])

        # To get b use all 0s
        b_circuit = Program()
        b_ro = b_circuit.declare('ro', 'BIT', len(self.qubits) + 1)
        b_circuit += bv_circuit
        b_circuit += [MEASURE(self.helper, b_ro[self.helper])]
        b_executable = qvm.compile(b_circuit)
        
        for i in range(self.n_trials):
            b_results = qvm.run(b_executable)
            print("trial {} b:".format(i))
            pprint.pprint(b_results)


# In[ ]:


if __name__ == '__main__':
    args=parser.parse_args()
    
    ## Assert bit string
    p = set(args.a)
    o = set(args.b)
    s = {'0','1'}
    if s == p or p == {'0'} or p == {'1'}:
    #     print('ok')
        pass
    else:
        raise AssertionError("a must be a bit string")
    if o == {'0'} or o == {'1'}:
    #     print('ok')
        pass
    else:
        raise AssertionError("b must be a bit")
    
    bit = create_bv_bitmap(args.a, args.b)
    with local_forest_runtime():
        bv = Bernstein_Vazirani()
        bv.run_bv(bit)

