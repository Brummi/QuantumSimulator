import numpy as np


class QuantumGate:
    def __init__(self, n):
        self.wires = []
        self.n = n
        self.op_mat = None

    def _build_naive_op_mat(self):
        raise NotImplementedError

    def apply(self, st_vec: np.ndarray):
        raise NotImplementedError

    def naive_apply(self, st_vec: np.ndarray):
        if self.op_mat is None: self._build_naive_op_mat()
        return self.op_mat @ st_vec

    def __len__(self):
        return len(self.wires)


class SingleQubitQuantumGate(QuantumGate):
    def __init__(self, wire, n, single_qubit_matrix):
        super().__init__(n)
        self.wires = [int(wire)]
        assert len(self.wires) == 1
        self.single_qubit_matrix = single_qubit_matrix
        assert self.single_qubit_matrix.shape == (2, 2)

    def _build_naive_op_mat(self):
        self.op_mat = np.eye(1, dtype=np.complex_)
        wire = self.wires[0]
        for i in range(self.n):
            if i != wire:
                self.op_mat = np.kron(np.eye(2, dtype=np.complex_), self.op_mat)
            else:
                self.op_mat = np.kron(self.single_qubit_matrix, self.op_mat)

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.reshape((2 ** (n - wire - 1), 2, 2 ** wire))
        new_st_vec = np.empty_like(st_vec)
        new_st_vec[:, 0, :] = st_vec[:, 0, :] * self.single_qubit_matrix[0, 0] + st_vec[:, 1, :] * self.single_qubit_matrix[0, 1]
        new_st_vec[:, 1, :] = st_vec[:, 0, :] * self.single_qubit_matrix[1, 0] + st_vec[:, 1, :] * self.single_qubit_matrix[1, 1]
        new_st_vec = new_st_vec.reshape((2 ** n))

        return new_st_vec


class PauliX(SingleQubitQuantumGate):
    def __init__(self, wire, n):
        super().__init__(wire, n, np.array([[0, 1], [1, 0]], dtype=np.complex_))

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.reshape((2 ** (n - wire - 1), 2, 2 ** wire))
        st_vec = st_vec[:, [1, 0], :]
        st_vec = st_vec.reshape((2 ** n))

        return st_vec


class PauliY(SingleQubitQuantumGate):
    def __init__(self, wires, n):
        super().__init__(wires, n, np.array([[0, -1j], [1j, 0]], dtype=np.complex_))
        self.multiplier = np.array([-1j, 1j], dtype=np.complex_).reshape((1, 2, 1))

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.reshape((2 ** (n - wire - 1), 2, 2 ** wire))
        st_vec = st_vec[:, [1, 0], :] * self.multiplier
        st_vec = st_vec.reshape((2 ** n))

        return st_vec


class PauliZ(SingleQubitQuantumGate):
    def __init__(self, wires, n):
        super().__init__(wires, n, np.array([[1, 0], [0, -1]], dtype=np.complex_))

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.copy().reshape((2 ** (n - wire - 1), 2, 2 ** wire))
        st_vec[:, 1, :] = st_vec[:, 1, :] * (-1)
        st_vec = st_vec.reshape((2 ** n))

        return st_vec


class Hadamard(SingleQubitQuantumGate):
    def __init__(self, wires, n):
        super().__init__(wires, n, (2 ** (-1/2)) * np.array([[1, 1], [1, -1]], dtype=np.complex_))

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.reshape((2 ** (n - wire - 1), 2, 2 ** wire))
        new_st_vec = np.empty_like(st_vec)
        new_st_vec[:, 0, :] = (st_vec[:, 0, :] + st_vec[:, 1, :]) * (2 ** (-1/2))
        new_st_vec[:, 1, :] = (st_vec[:, 0, :] - st_vec[:, 1, :]) * (2 ** (-1 / 2))
        new_st_vec = new_st_vec.reshape((2 ** n))

        return new_st_vec


class PhaseS(SingleQubitQuantumGate):
    def __init__(self, wires, n):
        super().__init__(wires, n, np.array([[1, 0], [0, 1j]], dtype=np.complex_))

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.copy().reshape((2 ** (n - wire - 1), 2, 2 ** wire))
        st_vec[:, 1, :] = st_vec[:, 1, :] * 1j
        st_vec = st_vec.reshape((2 ** n))

        return st_vec


class Pi8thT(SingleQubitQuantumGate):
    def __init__(self, wires, n):
        super().__init__(wires, n, np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex_))

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.copy().reshape((2 ** (n - wire - 1), 2, 2 ** wire))
        st_vec[:, 1, :] = st_vec[:, 1, :] * np.exp(1j * np.pi / 4)
        st_vec = st_vec.reshape((2 ** n))

        return st_vec


class RotX(SingleQubitQuantumGate):
    def __init__(self, wires, n, angle):
        super().__init__(wires, n, np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)], [-1j * np.sin(angle / 2), np.cos(angle / 2)]], dtype=np.complex_))


class RotY(SingleQubitQuantumGate):
    def __init__(self, wires, n, angle):
        super().__init__(wires, n, np.array([[np.cos(angle / 2), - np.sin(angle / 2)], [- np.sin(angle / 2), np.cos(angle / 2)]], dtype=np.complex_))


class RotZ(SingleQubitQuantumGate):
    def __init__(self, wires, n, angle):
        super().__init__(wires, n, np.array([[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]], dtype=np.complex_))
        self.multiplier = np.array([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)], dtype=np.complex_).reshape((1, 2, 1))

    def apply(self, st_vec: np.ndarray):
        wire = self.wires[0]
        n = self.n

        st_vec = st_vec.reshape((2 ** (n - wire - 1), 2, 2 ** wire))
        st_vec = st_vec * self.multiplier
        st_vec = st_vec.reshape((2 ** n))

        return st_vec

class CNOT(QuantumGate):
    def __init__(self, wires, target, n):
        super().__init__(n)
        self.wires = list(sorted(wires))
        self.target = target
        assert len(self.wires) == 2
        assert self.target in self.wires

    def _build_naive_op_mat(self):
        zero_mat = np.eye(1, dtype=np.complex_)
        one_mat = np.eye(1, dtype=np.complex_)
        for i in range(self.n):
            if i not in self.wires:
                zero_mat = np.kron(np.eye(2, dtype=np.complex_), zero_mat)
                one_mat = np.kron(np.eye(2, dtype=np.complex_), one_mat)
            else:
                if i != self.target:
                    zero_mat = np.kron(np.array([[1, 0], [0, 0]], dtype=np.complex_), zero_mat)
                    one_mat = np.kron(np.array([[0, 0], [0, 1]], dtype=np.complex_), one_mat)
                else:
                    zero_mat = np.kron(np.eye(2, dtype=np.complex_), zero_mat)
                    one_mat = np.kron(np.array([[0, 1], [1, 0]], dtype=np.complex_), one_mat)
        self.op_mat = zero_mat + one_mat

    def apply(self, st_vec: np.ndarray):
        n = self.n

        slices = []
        pref_wire = n
        for wire in reversed(self.wires):
            slices += [2 ** (pref_wire - wire - 1), 2]
            pref_wire = wire
        slices += [2 ** pref_wire]

        st_vec = st_vec.copy().reshape(tuple(slices))

        if self.target == self.wires[0]:
            st_vec[:, 1, :, :, :] = np.take(st_vec[:, 1, :, :, :], [1, 0], axis=2)
        else:
            st_vec[:, :, :, 1, :] = np.take(st_vec[:, :, :, 1, :], [1, 0], axis=1)

        st_vec = st_vec.reshape((2 ** n))
        return st_vec