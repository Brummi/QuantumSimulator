# Quantum Simulator

This project is a state-vector based quantum circuit simulator written in Python and NumPy by Felix Wimbauer for the seminar 
"Topics of Quantum Computing" at TUM by Prof. Mendl.

## Features

The idea behind this simulator is to allow users to compare a naive state-vector simulation approach with an optimized 
approach that uses smart indexing to apply quantum gates.

Every implemented gate has the following functions:

- ```_build_naive_op_mat()``` Computes the naive operator matrix and stores it. The naive operator matrix can be accessed 
through the ```op_mat``` attribute.
- ```apply(st_vec)``` Applies the gate efficiently on the state-vector and returns the new state-vector.
- ```naive_apply(st_vec)``` Applies the gate naively on the state-vector and returns the new state-vector. 
If the operator matrix has not been built already the function first calls ```_build_naive_op_mat()```.


The following gates are implemented:

- ```PauliX```, ```PauliX```, ```PauliX```
- ```Hadamard```
- ```PhaseS```, ```Pi8thT```
- ```RotX```, ```RotY```, ```RotZ```
- ```CNOT```

## Usage

Gates can be chained together through the ```circuit``` class, which applies the ```apply(st_vec)``` / 
```naive_apply(st_vec)``` on all gates sequentially.

A state-vector is always a numpy array of type ```np.complex_```.

The ```test.py``` script offers insights on the usage of selected single gates. The ```benchmark.py``` script compares the 
runtime of naive apply vs efficient apply for a circuit, that applies ```Hadamard``` for every qubit in a 10-qubit system.

## Inspiration

This simulator was inspired by https://github.com/Qaintum/Qaintessent.jl