# Optimization over bounded-rank matrices through a desingularization enables joint global and local guarantees 

This is the code to reproduce the experiments of the following paper:

> *Optimization over bounded-rank matrices through a desingularization enables joint global and local guarantees*
> 
> Quentin Rebjock and Nicolas Boumal
> 
> <https://arxiv.org/abs/2406.14211>

It requires a version of [Manopt](https://www.manopt.org/) at least as recent as June 20, 2024.

This is a matrix completion problem.
Given a matrix $A \in \mathbb{R}^{m \times n}$ to recover and an observation mask $\Omega \in \\{0, 1\\}^{m \times n}$, minimize
$$f(X) = \frac{1}{2}\|\|(X - A) \odot \Omega\|\|_\mathrm{F}^2$$
over the bounded-rank variety
$$\\{X \in \mathbb{R}^{m \times n} : \mathrm{rank} \\, X \leq r\\}.$$

We consider three geometries:
- the LR parameterization (`completionlr.m`);
- the fixed-rank manifold (`completionfixed.m`);
- the desingularization (`completiondesingularization.m`).

The files `experiment1.m`, `experiment2.m` and `experiment3.m` contain the three experiments.
