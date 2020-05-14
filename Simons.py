from pyquil import get_qc, Program
from pyquil.gates import CNOT, Z, X, H, I, MEASURE

import argparse
from pprint import pprint

import numpy as np
from collections import defaultdict
import random
from pprint import pprint


class Simons(object):
    def __init__(self, n, f):
        self.input_qbits = range(n)
        self.ancilla_qbits = range(n, 2 * n)
        self.qbits = list(self.input_qbits) + list(self.ancilla_qbits)
        self._oracle = self._create_oracle(f)

    def _create_oracle(self, f):
        n = len(list(f.keys())[0])
        U = np.zeros(shape=(2 ** (2 * n), 2 ** (2 * n)))
        for a in range(2 ** n):
            ab = np.binary_repr(a, n)
            for k, v in f.items():
                U[int(ab + k, 2), int(xor(ab, v) + k, 2)] = 1
        return U

    def _create_circuit(self, oracle):
        p = Program()
        classical_reg = p.declare('ro', 'BIT', len(self.input_qbits))
        p.defgate('oracle', oracle)
        p += [H(x) for x in self.input_qbits]
        p += tuple(['oracle'] + sorted(self.qbits, reverse=True))
        p += [H(x) for x in self.input_qbits]
        p += [MEASURE(x, y) for x, y in zip(self.input_qbits, classical_reg)]
        return p

    def _binary_gauss_solve(self, M, v):
        v_ = np.copy(v)
        n = len(v)
        for r in range(n - 2, -1, -1):
            row = M[r]
            for c in range(r + 1, n):
                if row[c] == 1:
                    v_[r] = int(v[r]) ^ int(v[c])
        return v_[::-1]

    def _sample_vectors(self, p, qvm):
        # Sample n - 1 orthogonal vectors by running the quantum circuit ~n times
        msb_to_vector = {}
        while len(msb_to_vector) < len(self.input_qbits) - 1:
            e = qvm.compile(p)
            vector = np.array(qvm.run(e)[0], dtype=int)

            # Only add the sampled vector to the dict if the most significant bit
            # (i.e. leading 1) is not already present
            if (vector == 0).all() or (vector == 1).all():
                continue
            msb = np.argmax(vector == 1)
            if msb not in msb_to_vector:
                print(f'Sampled vector: {vector}')
                msb_to_vector[msb] = vector
            else:
                existing = msb_to_vector[msb]
                # xored is guaranteed to be orthogonal
                xored = np.array([x ^ y for x, y in zip(existing, vector)])
                if (xored == 0).all():
                    continue
                msb_xored = np.argmax(xored == 1)
                if msb_xored not in msb_to_vector:
                    print(f'Sampled vector: {vector}')
                    msb_to_vector[msb_xored] = msb_to_vector
        return msb_to_vector

    def _add_missing_msb(self, msb_to_vector):
        # Find missing most significant bit and add the corresponding unit vector
        msb = None
        for i in range(len(self.input_qbits)):
            if i not in msb_to_vector.keys():
                msb = i
                break
        vector = np.zeros(shape=(len(self.input_qbits)))
        vector[msb] = 1
        msb_to_vector[msb] = vector
        return msb_to_vector, msb

    def run(self):
        qvm = get_qc('9q-square-qvm')
        qvm.compiler.client.timeout = 120
        p = self._create_circuit(self._oracle)
        msb_to_vector = self._sample_vectors(p, qvm)
        msb_to_vector, msb = self._add_missing_msb(msb_to_vector)

        # Solve equation Ms = v where M is the msb matrix and v is the missing msb unit vector
        mat = np.asarray([x[1] for x in sorted(msb_to_vector.items(), key=lambda x: x[0])])
        print('Solving matrix:')
        pprint(mat)
        target = np.zeros(shape=(len(self.input_qbits),), dtype=int)
        target[msb] = 1
        return list(self._binary_gauss_solve(mat, target))


def xor(x, y):
    assert len(x) == len(y)
    n = len(x)
    return format(int(x, 2) ^ int(y, 2), f'0{n}b')


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

    new_range = np.random.choice(list(sorted(mapping.keys())), replace=False, size=n).tolist()
    mapping_pairs = sorted([(k, v) for k, v in mapping.items()], key=lambda x: x[0])

    new_mapping = {}
    # f(x) = f(x xor s)
    for i in range(n):
        x = mapping_pairs[i]
        y = new_range[i]
        new_mapping[x[0]] = y
        new_mapping[x[1]] = y

    return new_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('string', help='Secret string s of length n')
    parser.add_argument('ftype', type=int, help='1 for one-to-one or 2 for two-to-one')
    args = parser.parse_args()

    assert all([x == '1' or x == '0' for x in args.string]), 'string argument must be a binary string.'

    n = len(args.string)
    if args.ftype == 1:
        mapping = one_to_one_mapping(args.string)
    elif args.ftype == 2:
        mapping = two_to_one_mapping(args.string)
    else:
        raise ValueError('Invalid function type.')

    print('Generated mapping:')
    pprint(mapping)
    simons = Simons(n, mapping)
    result = simons.run()
    result = ''.join([str(x) for x in result])
    # Check if result satisfies two-to-one function constraint
    success = np.array([mapping[x] == mapping[xor(x, result)] for x in mapping.keys()]).all()
    if success:
        print(f'Oracle function is two-to-one with s = {result}.')
    else:
        print('Oracle is one-to-one.')


if __name__ == '__main__':
    main()