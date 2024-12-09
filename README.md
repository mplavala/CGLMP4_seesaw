# Code to accompany: Experimental implementation of dimension-dependent contextuality inequality

This repository contains Python code for the see-saw maximization of the violation of the CGLMP<sub>4</sub> contextuality inequality. Optimization runs over arbitrary measurements on two qubits, and over preparations of separable two qubit states.

Running the see-saw optimization algorithm `N = 10000` took approximately 10 hours on a personal computer.

Numerical results of running the algorithm are also included.

## The see-saw optimization algorithm

1. Two random projection-valued measures (PVMs) are generated from eigenvectors of random hermitian operators.
2. Given the two measurements (in general POVMs), the separable preparations (assemblages) maximizing the violation of the CGLMP<sub>4</sub> contextuality inequality are found by solving the respective SDP. Separability is implemented by requiring that the respective density matrices have positive partial transpose.
3. Given the two assemblages, the measurements (POVMs) maximizing the violation of the CGLMP<sub>4</sub> contextuality inequality are found by solving the respective SDP.
4. Steps 2. and 3. are repeated until the same maximal violation is obtained 10 times in a row (with precision to 4 decimal places).

## Contents

- `seesaw.py`: Python code that performs the optimization. See-saw optimization is performed `N` times, by default `N = 10000` is set.
- `results-N-10000-I_4-0.336085803661498.npy` data from running the see-saw optimization 10 000 times with highest violation of 0.336085803661498.
- `results-N-100-I_4-0.33606808585730574.npy` data from running the see-saw optimization 100 times with highest violation of 0.33606808585730574.
- `results-N-10-I_4-0.3360675301596743.npy` data from running the see-saw optimization 10 times with highest violation of 0.3360675301596743.
