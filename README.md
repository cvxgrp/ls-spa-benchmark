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

In addition, the medium experiment requires the library implementation of `ls_spa`
found at [cvxgrp/ls-spa](https://github.com/cvxgrp/ls-spa).

JAX is a dependency of `ls_spa`, but its installation varies by platform (do not try to 
blindly `pip install jax`). Follow 
[these instructions](https://github.com/google/jax#installation) to correctly install JAX.
The other packages are safely `pip` installable.

To run the benchmark code, clone this repository and install the dependencies. Afterwards, 
the two experiment files can be executed with Python.

If you use this code for research, please cite the associated paper.

```bibtex
@article{Bell2024,
  title = {Efficient Shapley performance attribution for least-squares regression},
  volume = {34},
  ISSN = {1573-1375},
  url = {http://dx.doi.org/10.1007/s11222-024-10459-9},
  DOI = {10.1007/s11222-024-10459-9},
  number = {5},
  journal = {Statistics and Computing},
  publisher = {Springer Science and Business Media LLC},
  author = {Bell,  Logan and Devanathan,  Nikhil and Boyd,  Stephen},
  year = {2024},
  month = jul 
}
```
