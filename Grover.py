#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
parser.add_argument('--bit', help='Enter the bit string for your function to search for')
parser.add_argument('--valid', help='Enter valid for valid grover mapping or invalid for no string returning 1')


# In[5]:


def create_grover_valid_bitmap(bit):
    n_bits = len(bit)
    bit_map = {}
    for bit_val in range(2 ** n_bits):
        if(np.binary_repr(bit_val, width=n_bits) == bit):
            bit_map[np.binary_repr(bit_val, width=n_bits)] = str(1)
        else:
            bit_map[np.binary_repr(bit_val, width=n_bits)] = str(0)

    return bit_map

def create_grover_invalid_bitmap(bit):
    n_bits = len(bit)
    bit_map = {}
    for bit_val in range(2 ** n_bits):
        bit_map[np.binary_repr(bit_val, width=n_bits)] = str(0)

    return bit_map

class Grover(object):
    def _init_(self):        
        self.n_qubits = None
        self.qubits = None
        self.bit_map = None
        self.num_iter = None
            
    def _run_init(self, bit):
        self.bit_map = bit
        self.n_qubits = 2**len(list(bit.keys())[0])
        self.qubits = list(range(int(np.log2(self.n_qubits))))
        self.num_iter = int(round(np.pi * 2 ** (len(self.qubits) / 2.0 - 2.0)))

    def _grover_oracle_matrix(self, bit):
        n_bits = len(list(bit.keys())[0])
        oracle_matrix = np.zeros(shape=(2 ** n_bits, 2 ** n_bits))
        for b in range(2 ** n_bits):
            bin_str = np.binary_repr(b, n_bits)
            fill_value = bit[bin_str]
            if fill_value == '0': fill_value = '-1'
            oracle_matrix[b, b] = fill_value
        return oracle_matrix
    
    def _grover_diffusion_op(self):
        diffusion_program = Program()
        dim = 2 ** len(self.qubits)
        diffusion_matrix = np.diag([1.0] + [-1.0] * (dim - 1))
        diffusion_program.defgate('diffusion', diffusion_matrix)
        instruction_tuple = ('diffusion',) + tuple(self.qubits)
        diffusion_program.inst(instruction_tuple)
        return diffusion_program
        
    def grover_run(self, bit):
        self._run_init(bit)
    
        oracle = Program()
        oracle_name = "grover_oracle"
        oracle.defgate(oracle_name, self._grover_oracle_matrix(bit))
        oracle.inst(tuple([oracle_name] + self.qubits))
    
        diffusion = self._grover_diffusion_op()

        p = Program()
        Hm = Program().inst([H(qubit) for qubit in grover.qubits])
        p += Hm
        ## Repeating part of the algorithm
        for _ in range(self.num_iter):
            p += oracle
            p += Hm
            p += diffusion 
            p += Hm
            
        # run the program on a QVM
        qc = get_qc('9q-square-qvm')
        result = qc.run_and_measure(p, trials=10)
        pprint.pprint(result)


# In[6]:


if __name__ == '__main__':
    args=parser.parse_args()
    
    ## Assert bit string
    p = set(args.bit)
    s = {'0','1'}
    if s == p or p == {'0'} or p == {'1'}:
    #     print('ok')
        pass
    else:
        raise AssertionError("bit must be a bit strings")
        
        
    if args.valid == 'valid':
        bit = create_grover_valid_bitmap(args.bit)
    elif args.valid == 'invalid':
        bit = create_grover_invalid_bitmap(args.bit)
    else:
        raise AssertionError("valid must be a either 'valid' or 'invalid' strings")
        
    with local_forest_runtime():
        grover = Grover()
        grover.grover_run(bit)


# In[ ]:




