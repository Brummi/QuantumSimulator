import numpy as np
from logic.gates import *

# Wire 0 is right-most wire
cnot = CNOT(wires=[0, 2], target=0, n=3)
x = PauliX(wires=[2], n=3)

cnot._build_naive_op_mat()
x._build_naive_op_mat()

v = np.array([1, 2, 3, 4, 5, 6, 7, 8])

cnot_v = cnot.apply(v)
cnot_v_naive = cnot.naive_apply(v)

x_v = x.apply(v)
x_v_naive = x.naive_apply(v)

print(f"v: {v}")
print(f"cnot_v: {cnot_v}")
print(f"cnot_v_naive: {cnot_v_naive.real}")
print(f"x_v: {x_v}")
print(f"x_v_naive: {x_v_naive.real}")

