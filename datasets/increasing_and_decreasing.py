"""
A harder version of increasing.
Classes:
[1,0,0] -> not monotonic increasing sequence
[0,1,0] -> monotonic increasing sequence
[0,0,1] -> monotonic decreasing sequence
"""
import random
from tqdm import tqdm


MAX_ITER = 1_000  # max iter when trying to generate new sequence


class IncreasingDecreasingDataset:

    def __init__(self, k, n_train=1_000_000, n_valid=0, n_test=0):
        self.k = k  # length of sequences
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.train = {}
        self.valid = {}
        self.test = {}

        print("[increasing_and_decreasing.py] Preparing valid.")
        for _ in tqdm(range(self.n_valid // 3)):
            ordered_sequence = tuple(sorted([random.randint(-100, 100) for _ in range(self.k)]))
            if ordered_sequence not in self.valid:
                self.valid[ordered_sequence] = [0, 1, 0]
        for _ in tqdm(range(self.n_valid // 3)):
            ordered_sequence = tuple(reversed(sorted([random.randint(-100, 100) for _ in range(self.k)])))
            if ordered_sequence not in self.valid:
                self.valid[ordered_sequence] = [0, 0, 1]
        for _ in tqdm(range(self.n_valid - 2 * (self.n_valid // 3))):
            sequence = tuple([random.randint(-100, 100) for _ in range(self.k)])
            if sequence == tuple(sorted(sequence)) or sequence == tuple(reversed(sorted(sequence))):
                continue
            if sequence not in self.valid:
                self.valid[sequence] = [1, 0, 0]

        print("[increasing_and_decreasing.py] Preparing test.")
        for _ in tqdm(range(self.n_test // 3)):
            ordered_sequence = tuple(sorted([random.randint(-100, 100) for _ in range(self.k)]))
            if ordered_sequence not in self.test and ordered_sequence not in self.valid:
                self.test[ordered_sequence] = [0, 1, 0]
        for _ in tqdm(range(self.n_test // 3)):
            ordered_sequence = tuple(reversed(sorted([random.randint(-100, 100) for _ in range(self.k)])))
            if ordered_sequence not in self.test and ordered_sequence not in self.valid:
                self.test[ordered_sequence] = [0, 0, 1]
        for _ in tqdm(range(self.n_test - 2 * (self.n_test // 3))):
            sequence = tuple([random.randint(-100, 100) for _ in range(self.k)])
            if sequence == tuple(sorted(sequence)) or sequence == tuple(reversed(sorted(sequence))):
                continue
            if sequence not in self.test and sequence not in self.valid:
                self.test[sequence] = [1, 0, 0]

    def train_iterator(self):
        # NOTICE: We allow repeating elements of training!
        for _ in range(self.n_train):
            choice = random.randint(0, 2)
            if choice == 1:
                for _ in range(MAX_ITER):
                    sequence = tuple(sorted([random.randint(-100, 100) for _ in range(self.k)]))
                    if sequence not in self.test and sequence not in self.valid:
                        yield sequence, [0, 1, 0]
                        break
                else:
                    raise Exception("[increasing_and_decreasing.py] Reached maximum number of iterations")
            elif choice == 2:
                for _ in range(MAX_ITER):
                    sequence = tuple(reversed(sorted([random.randint(-100, 100) for _ in range(self.k)])))
                    if sequence not in self.test and sequence not in self.valid:
                        yield sequence, [0, 0, 1]
                        break
                else:
                    raise Exception("[increasing_and_decreasing.py] Reached maximum number of iterations")
            else:
                for _ in range(MAX_ITER):
                    sequence = tuple([random.randint(-100, 100) for _ in range(self.k)])
                    if sequence == tuple(sorted(sequence)):
                        continue
                    if sequence not in self.test and sequence not in self.valid:
                        yield sequence, [1, 0, 0]
                        break
                else:
                    raise Exception("[increasing_and_decreasing.py] Reached maximum number of iterations")

    def valid_iterator(self):
        for sequence in self.valid:
            yield sequence, self.valid[sequence]

    def test_iterator(self):
        for sequence in self.test:
            yield sequence, self.test[sequence]

