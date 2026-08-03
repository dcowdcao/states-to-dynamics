"""Where does the regularizer start to bite? Probe the MSE / gamma*reg crossover.

Short smoke-length runs across a wide gamma range, reporting the two loss terms. The
crossover gamma is where a sparsity sweep should be centered. For CosineSpaDE the
regularizer is bounded in [0, 2] by the simplex while the MSE term is order 10, which puts
parity near gamma = 16 — three orders of magnitude above the default.
"""

# def main(): ...   # --sweep_dir -> per-gamma loss terms + the parity estimate
