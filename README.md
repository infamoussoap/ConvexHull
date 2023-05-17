# Convex optimization over a probability simplex
This repository contains the code for our paper [Convex optimization over a probability simplex](https://arxiv.org/abs/2305.09046). It has usable code for <ins>Projection onto a Convex Hull</ins>, <ins>Optimal Question Weighting</ins>, and reproducible code for these sections in the paper.

Our other [repository](https://github.com/infamoussoap/UniversalPortfolio) contains the code for <ins>Universal Portfolios</ins>.

## Projection onto a Convex Hull
Let $(x_i)_{1\leq i \leq N}$ be a set of points with $x_i\in\mathbb{R}^d$. For some $y\in\mathbb{R}^d$, projection onto a convex hull involves solving the minimization problem
$$\min_w \|\|wX - y \|\|^2\quad \text{where}\quad \sum_i w_i = 1\ \text{and}\ w_i\geq0,$$
and $X=[x_1,\ldots, x_N]^T$ is a $N\times d$ matrix. This is also known as simplex-constrained regression.

While quadratic programs can solve this problem, they are often slow when $n$ or $d$ is big. In our code, we propose a new iteration algorithm to solve this problem.

More details can be found in our paper.

<b>Implemented Algorithms</b>: We implement Cauchy-Simplex, Pariwise Frank-Wolfe, and Exponentiated Gradient Descent. 

## Optimal Question Weighting
It is often desirable that the distribution of exam marks matches a target distribution, but this rarely happens. However, altering the weights of each question will alter the final distribution.

Our code proposes a new algorithm to find such weights that the final distribution will match a target distribution. An example can be seen in the picture below.

More details can be found in our paper.

<b>Implemented Algorithms</b>: We implement Cauchy-Simplex, Pariwise Frank-Wolfe, and Exponentiated Gradient Descent. 

![alt text](https://github.com/infamoussoap/ConvexHull/blob/main/Results/mark_distribution.png)
