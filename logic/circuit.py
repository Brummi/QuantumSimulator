import numpy as np


class Circuit:
    def __init__(self, gates: list, n: int):
        self.gates = gates
        self.n = n
        for gate in self.gates:
            assert len(gate) == self.n

    def _build_naive_op_mat(self):
        for gate in self.gates:
            gate._build_naive_op_mat()

    def apply(self, state_vector: np.ndarray):
        for gate in self.gates:
            state_vector = gate.apply(state_vector)
        return state_vector