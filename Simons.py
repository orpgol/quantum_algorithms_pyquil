from pyquil import get_qc, Program
from pyquil.gates import CNOT, Z, X, H, I, MEASURE

import argparse
from pprint import pprint

import numpy as np
from collections import defaultdict
import random
from pprint import pprint


class Simons(object):
    def __init__(self, n):
        self.input_qbits = range(n)
        self.ancilla_qbits = range(n, 2 * n)
        self.qbits = list(self.input_qbits) + list(self.ancilla_qbits)

    def run(self, oracle):
        qvm = get_qc('16q-qvm')
        qvm.compiler.client.timeout = 120
        
        # Sample n - 1 orthogonal vectors
        msb_to_vector = {}
        while len(msb_to_vector) < len(self.input_qbits) - 1:
            p = Program()
            classical_reg = p.declare('ro', 'BIT', len(self.input_qbits))
            p.defgate('oracle', oracle)
            p += [H(x) for x in self.input_qbits]
            p += tuple(['oracle'] + sorted(self.qbits, reverse=True))
            p += [H(x) for x in self.input_qbits]
            p += [MEASURE(x, y) for x, y in zip(self.input_qbits, classical_reg)]
            e = qvm.compile(p)
            vector = np.array(qvm.run(e)[0], dtype=int)

            if (vector == 0).all() or (vector == 1).all():
                continue
            msb = np.argmax(vector == 1)
            if msb not in msb_to_vector:
                print(f'Sampled vector: {vector}')
                msb_to_vector[msb] = vector
            else:
                existing = msb_to_vector[msb]
                # guaranteed to be orthogonal
                xored = np.array([x ^ y for x, y in zip(existing, vector)])
                if (xored == 0).all():
                    continue
                msb_xored = np.argmax(xored == 1)
                if msb_xored not in msb_to_vector:
                    print(f'Sampled vector: {vector}')
                    msb_to_vector[msb_xored] = msb_to_vector

        # Find missing most significant bit and augment with corresponding unit vector
        msb = None
        for i in range(len(self.input_qbits)):
            if i not in msb_to_vector.keys():
                msb = i
                break
        assert msb is not None, 'No missing MSB'

        vector = np.zeros(shape=(len(self.input_qbits)))
        vector[msb] = 1

        msb_to_vector[msb] = vector

        # Solve equation such that inner product of missing msb unit vector and s is 1
        mat = np.asarray([x[1] for x in sorted(msb_to_vector.items(), key=lambda x: x[0])])
        print('Solving matrix:')
        pprint(mat)
        target = np.zeros(shape=(len(self.input_qbits),), dtype=int)
        target[msb] = 1
        return list(binary_gauss_solve(mat, target))


def xor(x, y):
    assert len(x) == len(y)
    n = len(x)
    return format(int(x, 2) ^ int(y, 2), f'0{n}b')


def binary_gauss_solve(M, v):
    v_ = np.copy(v)
    n = len(v)
    for r in range(n - 2, -1, -1):
        row = M[r]
        for c in range(r + 1, n):
            if row[c] == 1:
                v_[r] = int(v[r]) ^ int(v[c])

    return v_[::-1]


def one_to_one_mapping(s):
    n = len(s)
    form_string = "{0:0" + str(n) + "b}"
    bit_map_dct = {}
    for idx in range(2 ** n):
        bit_string = np.binary_repr(idx, n)
        bit_map_dct[bit_string] = xor(bit_string, s)
    return bit_map_dct


def two_to_one_mapping(s):
    mapping = one_to_one_mapping(s)
    n = len(mapping.keys()) // 2

    new_range = list(np.random.choice(list(sorted(mapping.keys())), replace=False, size=n))
    mapping_pairs = sorted([(k, v) for k, v in mapping.items()], key=lambda x: x[0])

    new_mapping = {}
    # f(x) = f(x xor s)
    for i in range(n):
        x = mapping_pairs[i]
        y = new_range[i]
        new_mapping[x[0]] = y
        new_mapping[x[1]] = y

    return new_mapping


def create_oracle(mapping):
    n = len(list(mapping.keys())[0])
    U = np.zeros(shape=(2 ** (2 * n), 2 ** (2 * n)))
    for a in range(2 ** n):
        ab = np.binary_repr(a, n)
        for k, v in mapping.items():
            U[int(ab + k, 2), int(xor(ab, v) + k, 2)] = 1
    return U


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('string', help='Secret string s of length n')
    parser.add_argument('ftype', type=int, help='1 for one-to-one or 2 for two-to-one')
    args = parser.parse_args()

    n = len(args.string)
    if args.ftype == 1:
        mapping = one_to_one_mapping(args.string)
    elif args.ftype == 2:
        mapping = two_to_one_mapping(args.string)
    else:
        raise ValueError('Invalid function type.')
    print('Generated mapping:')
    pprint(mapping)
    oracle = create_oracle(mapping)
    simons = Simons(n)
    result = simons.run(oracle)
    result = ''.join([str(x) for x in result])
    success = np.array([mapping[x] == mapping[xor(x, result)] for x in mapping.keys()]).all()
    if success:
        print(f'Oracle function is two-to-one with s = {result}.')
    else:
        print('Oracle is one-to-one.')


if __name__ == '__main__':
    main()