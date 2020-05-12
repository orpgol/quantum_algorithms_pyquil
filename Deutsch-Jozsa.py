#!/usr/bin/env python
# coding: utf-8

# In[13]:


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
parser.add_argument('bits', help='Enter the number of bits you want to run with')
parser.add_argument('algo', help='Enter balanced for balanced or const for constant function')


# In[14]:

## Class representation of DJ algorithm
class Deutsch_Jozsa(object):
    def _init_(self):
        self.n_qubits = None
        self.n_helper = None
        self.qubits = None
        self.helper = None
        self.const = None
        self.balanced = None
        
    ## Run initialization on parameters needed for the algorithm
    def _run_init(self, n):
        self.n_qubits = n
        self.n_helper = 1
        self.qubits = list(range(self.n_qubits))
        self.helper = self.n_qubits

        self.const, self.balanced = self._create_oracle()
        
    ## Create oracle based on arbitrary number of bits n
    def _create_oracle(self):

        p_const = [I(qubit) for qubit in self.qubits + [self.helper]]
        p_balanced = [Z(qubit) for qubit in self.qubits + [self.helper]]

        return p_const, p_balanced
    
    ## Start an algorithm run
    def run_dj(self, n, balanced = True):
        self._run_init(n)
        qvm = get_qc('9q-square-qvm')
        typ = ''
        p = Program()
        p += X(self.helper)
        p += [H(qubit) for qubit in self.qubits]
        p += H(self.n_qubits)
        if balanced:
            typ = 'BALANCED'
            p += self.balanced
        else:
            TYP = 'CONST'
            p += self.const
        p += [H(qubit) for qubit in self.qubits]

        results = qvm.run_and_measure(p, trials=10)

        print(typ)
        pprint.pprint(results)


# In[15]:

## Command line interface with our algotrithm
if __name__ == '__main__':
    args=parser.parse_args()

    with local_forest_runtime():
        dj = Deutsch_Jozsa()
        bool = True
        if args.algo == 'const':
            bool = False
        dj.run_dj(int(args.bits), balanced=bool)


# In[ ]:
