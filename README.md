Benchmark code for the paper [Efficient Shapley Performance Attribution for Least-Squares
Regression](https://web.stanford.edu/~boyd/papers/ls_shapley.html) by Logan Bell,
Nikhil Devanathan, and Stephen Boyd. This code was written with primarily performance in mind. 
A more elegant, easy-to-use (but slightly less performant) library implementation of the 
reference paper can be found at [cvxgrp/ls-spa](https://github.com/cvxgrp/ls-spa).
We recommend using the library implementaton.

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

If you use this code for research, please cite the associated paper.
```
@misc{https://doi.org/10.48550/arxiv.2310.19245,
  doi = {10.48550/ARXIV.2310.19245},
  url = {https://arxiv.org/abs/2310.19245},
  author = {Bell,  Logan and Devanathan,  Nikhil and Boyd,  Stephen},
  keywords = {Computation (stat.CO),  FOS: Computer and information sciences,  FOS: Computer and information sciences,  62-08 (Primary),  62-04,  62J99 (Secondary)},
  title = {Efficient Shapley Performance Attribution for Least-Squares Regression},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual,  non-exclusive license}
}
```
