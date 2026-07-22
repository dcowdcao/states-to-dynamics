# SpaDE vs Karvonen

> Port from the original `docs/Spade_vs_Karvonen_Comparison.md`.

Documents that the pipeline runs on Karvonen's exact training/eval infrastructure (streaming
buffer, single `trainSAE()` loop, dict 8192, ~36,621 steps, turns 0-59, Karvonen's trainer
classes, both frac_recovered and FVE), and the genuine remaining differences: the novel
SpaDE/CosineSpaDE architectures, a small fixed set of Pareto configs vs Karvonen's ~240-config
sweep, and an R² formula nuance. No JumpReLU in this pipeline.
