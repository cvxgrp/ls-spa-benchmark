Benchmark code for the paper [Efficient Shapley Performance Attribution for Least-Squares
Regression](https://web.stanford.edu/~boyd/papers/ls_shapley.html) by Logan Bell,
Nikhil Devanathan, and Stephen Boyd. A more elegant (but slightly less performant) library 
implementation of the reference paper can be found at 
[cvxgrp/ls-spa](https://web.stanford.edu/~boyd/papers/ls_shapley.html).

The code has the following dependencies:
- `numpy`
- `scipy`
- `pandas`
- `jax`
- `matplotlib`

JAX is a dependency of `ls_spa`, but its installation varies by platform (do not try to 
blindly `pip install jax`). Follow 
[these instructions](https://github.com/google/jax#installation) to correctly install JAX.
The other packages are safely `pip` installable.

To run the benchmark code, clone this repository, add an empty directory called `plots`
to the root directory of the repository, and install the dependencies. Afterwards, the 
two experiment files can be executed with Python. The medium experiment generates Figure 2
in the companion paper, and the large experiment generates Figure 3 in the companion paper.
