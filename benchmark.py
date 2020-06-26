import timeit

qubits = 10

setup_optimized = f'''
import numpy as np
from logic.gates import Hadamard
from logic.circuit import Circuit

def setup_optimized():
    global c, v
    qubits = {qubits}
    gates = [Hadamard(i, qubits) for i in range(qubits)]
    c = Circuit(gates, qubits)
    v = np.zeros((2 ** qubits), dtype=np.complex_)
    v[0] = 1

setup_optimized()
'''

setup_naive = f'''
import numpy as np
from logic.gates import Hadamard
from logic.circuit import Circuit

def setup_naive():
    global c, v
    qubits = {qubits}
    gates = [Hadamard(i, qubits) for i in range(qubits)]
    c = Circuit(gates, qubits)
    c._build_naive_op_mat()
    v = np.zeros((2 ** qubits), dtype=np.complex_)
    v[0] = 1

setup_naive()
'''


testcode_optimized = "c.apply(v)"
testcode_naive = "c.naive_apply(v)"

print("Beginning optimized run")

optimized_time = timeit.timeit(stmt=testcode_optimized, setup=setup_optimized, number=1000)

print(f"Optimized time: {optimized_time:.5f}")

print("Beginning naive run")

naive_time = timeit.timeit(stmt=testcode_naive, setup=setup_naive, number=1000)

print(f"Naive time: {naive_time:.5f}")
