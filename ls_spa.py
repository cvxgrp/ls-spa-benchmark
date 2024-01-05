# Copyright 2024 Logan Bell, Nikhil Devanathan, and Stephen Boyd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains a method to efficiently estimate a Shapley 
attribution for least squares problems.

This method is described in the paper Efficient Shapley Performance 
Attribution for Least-Squares Regression (arXiv:2310.19245) by Logan 
Bell, Nikhil Devanathan, and Stephen Boyd.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Literal, Tuple

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import jit, random, vmap
import pandas as pd
from scipy.stats.qmc import MultivariateNormalQMC, Sobol

SampleMethod = Literal['random', 'permutohedron', 'argsort', 'exact']

@dataclass
class ShapleyResults:
    """
    A dataclass to store the results of the Shapley attribution.

    Attributes:
        attribution (jnp.ndarray): The Shapley attribution.
        attribution_history (jnp.ndarray | None): The Shapley attribution
            at each iteration of the algorithm.
        theta (jnp.ndarray): The coefficients of the linear model.
        overall_error (float): The overall error of the Shapley attribution.
        error_history (jnp.ndarray | None): The overall error of the Shapley
            attribution at each iteration of the algorithm.
        attribution_errors (jnp.ndarray): The error of each feature in the
            Shapley attribution.
        r_squared (float): The R^2 of the linear model.
    """
    attribution: jnp.ndarray
    attribution_history: jnp.ndarray | None
    theta: jnp.ndarray
    overall_error: float
    error_history: jnp.ndarray | None
    attribution_errors: jnp.ndarray
    r_squared: float

    def __repr__(self):
        """Makes printing the dataclass look nice."""

        attr_str = "(" + "".join("{:.2f}, ".format(a) for a in self.attribution.flatten())[:-2] + ")"
        coefs_str = "(" + "".join("{:.2f}, ".format(c) for c in self.theta.flatten())[:-2] + ")"

        return """
        p = {}
        Out-of-sample R^2 with all features: {:.2f}

        Shapley attribution: {}
        Estimated error in Shapley attribution: {:.2E}

        Fitted coeficients with all features: {}
        """.format(
            len(self.attribution.flatten()),
            self.r_squared,
            attr_str,
            self.overall_error,
            coefs_str
        )


class SizeIncompatible(Exception):
    """Custom exception for incompatible data sizes."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def validate_data(X_train: jnp.ndarray,
                  X_test: jnp.ndarray,
                  y_train: jnp.ndarray,
                  y_test: jnp.ndarray):
    """
    Validates the data for the Shapley attribution.

    Args:
        X_train (jnp.ndarray): The training data.
        X_test (jnp.ndarray): The test data.
        y_train (jnp.ndarray): The training labels.
        y_test (jnp.ndarray): The test labels.

    Raises:
        SizeIncompatible: If the data is not compatible.
    """
    
    if X_train.shape[1] != X_test.shape[1]:
        raise SizeIncompatible("X_train and X_test should have the "
                               "same number of columns (features).")

    if X_train.shape[0] != y_train.shape[0]:
        raise SizeIncompatible("X_train should have the same number of "
                               "rows as y_train has entries (observations).")

    if X_test.shape[0] != y_test.shape[0]:
        raise SizeIncompatible("X_test should have the same number of "
                               "rows as y_test has entries (observations).")

    if X_train.shape[1] > X_train.shape[0]:
        raise SizeIncompatible("The function works only if the number of "
                               "features is at most the number of "
                               "observations.")


def ls_spa(X_train: np.ndarray | jnp.ndarray | pd.DataFrame,
           X_test: np.ndarray | jnp.ndarray | pd.DataFrame,
           y_train: np.ndarray | jnp.ndarray | pd.Series,
           y_test: np.ndarray | jnp.ndarray | pd.Series,
           reg: float = 0.,
           method: SampleMethod | None = None,
           batch_size: int = 2 ** 7,
           num_batches: int = 2 ** 7,
           tolerance: float = 1e-2,
           seed: int = 42) -> ShapleyResults:
    """
    Computes the Shapley attribution for a least squares problem.

    Args:
        X_train (np.ndarray | jnp.ndarray | pd.DataFrame): The training data.
        X_test (np.ndarray | jnp.ndarray | pd.DataFrame): The test data.
        y_train (np.ndarray | jnp.ndarray | pd.Series): The training labels.
        y_test (np.ndarray | jnp.ndarray | pd.Series): The test labels.
        reg (float, optional): The regularization parameter. Defaults to 0..
        method (SampleMethod | None, optional): The sampling method to use.
            Defaults to None.
        batch_size (int, optional): The batch size. Defaults to 2 ** 7.
        num_batches (int, optional): The maximum number of batches to use.
            Defaults to 2 ** 7.
        tolerance (float, optional): The tolerance for the algorithm to
            terminate. Defaults to 1e-2.
        seed (int, optional): The seed for the random number generator.
            Defaults to 42.

    Returns:
        ShapleyResults: The results of the Shapley attribution.
    """
    # Converting data into JAX arrays.
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    validate_data(X_train, X_test, y_train, y_test)

    N, p = X_train.shape
    M, _ = X_test.shape
    if method is None:
        if p > 10:
            method = 'argsort'
        else:
            method = 'argsort' ### XXX exact needs to be implemented still

    rng = random.PRNGKey(seed)
    compute_spa = LSSPA(key=rng,
                       p=p,
                       sample_method=method,
                       batch_size=batch_size)

    return compute_spa(X_train=X_train,
                       X_test=X_test,
                       y_train=y_train,
                       y_test=y_test,
                       reg=reg,
                       max_num_batches=num_batches,
                       eps=tolerance,
                       return_history=False)


class Permutations(ABC):
    """
    An abstract class for generating permutations.

    Attributes:
        key (jax.random.PRNGKey): The random number generator key.
        p (int): The number of features.
    """

    def __init__(self, key, p: int):
        """
        Args:
            key (jax.random.PRNGKey): The random number generator key.
            p (int): The number of features.
        """
        self.key = key
        self.p = p

    @abstractmethod
    def __call__(self, num_perms: int) -> jnp.ndarray:
        """
        Args:
            num_perms (int): The number of permutations to generate.

        Returns:
            jnp.ndarray: The generated permutations.
        """
        pass

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        self._p = new_p


class RandomPermutations(Permutations):
    """
    A class for generating random permutations via Monte Carlo sampling.
    """

    def __call__(self, num_perms: int) -> jnp.ndarray:
        # Split the key to ensure different permutations each call
        self.key, keygenkey = random.split(self.key)
        to_permute =jnp.tile(jnp.arange(0, self.p), (num_perms, 1))
        # Generate random permutations
        return random.permutation(keygenkey, to_permute,
                                  axis=1, independent=True)


class PermutohedronPermutations(Permutations):
    """
    A class for generating permutations via the permutohedron sampling method.
    """

    def __init__(self, key, p: int):
        self.key = key
        self.p = p

    def __call__(self, num_perms: int) -> jnp.ndarray:
        # Generate permutohedron samples
        samples = jnp.array(self.qmc.random(num_perms))
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        samples = self.project(samples)
        samples = jnp.argsort(samples, axis=1)
        return samples

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        self._p = new_p
        self.key, keygenkey = random.split(self.key)
        seed = int(random.choice(keygenkey, 100000))
        self.qmc = MultivariateNormalQMC(np.zeros(self.p-1), seed=seed,
                                         inv_transform=False)

    @partial(jit, static_argnums=0)
    def project(self, x: jnp.ndarray):
        """
        A helper function to project samples onto the permutohedron.

        Args:
            x (jnp.ndarray): The samples to project.

        Returns:
            jnp.ndarray: The projected samples.
        """
        tril_part = jnp.tril(jnp.ones((self.p-1, self.p)))
        diag_part = jnp.diag(-jnp.arange(1, self.p), 1)[:-1]
        U = tril_part + diag_part
        U = U / jnp.linalg.norm(U, axis=1, keepdims=True)
        return x @ U


class ArgsortPermutations(Permutations):
    """
    A class for generating permutations via the argsort sampling method.
    """

    def __init__(self, key, p: int):
        self.key = key
        self.p = p

    def __call__(self, num_perms: int) -> jnp.ndarray:
        # Generate argsort samples
        samples = jnp.array(self.qmc.random(num_perms))
        return jnp.argsort(samples, axis=1)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        self._p = new_p
        self.key, keygenkey = random.split(self.key)
        seed = int(random.choice(keygenkey, 100000))
        self.qmc = Sobol(self.p, seed=seed)


class RiskEstimate:
    """
    A class for estimating the risk of the Shapley attribution.

    Attributes:
        key (jax.random.PRNGKey): The random number generator key.
        batch_size (int): The batch size.
        p (int): The number of features.
        mean (jnp.ndarray): The mean of the Shapley attribution.
        cov (jnp.ndarray): The covariance of the Shapley attribution.
        _i (int): The number of batches seen so far.
    """

    def __init__(self, key, batch_size: int, p: int):
        self.key = key
        self.batch_size = batch_size
        self.p = p
        self.mean = jnp.zeros((self.p,))
        self.cov = jnp.zeros((self.p, self.p))
        self._i = 1

        def risk_sample(key, cov):
            """
            A helper function to sample from the risk distribution.

            Args:
                key (jax.random.PRNGKey): The random number generator key.
                cov (jnp.ndarray): The covariance of the Shapley attribution.

            Returns:
                Tuple[jnp.ndarray, float]: The 95th percentile of the
                    absolute value of the feature attribution and the 95th
                    percentile of the norm of the feature attribution.
            """
            sample_diffs = random.multivariate_normal(key, jnp.zeros(p),
                                                      cov, shape=((1000,)),
                                                      method='svd')
            abs_diffs = jnp.abs(sample_diffs)
            norms = jnp.linalg.norm(sample_diffs, axis=1)
            abs_quantile = jnp.quantile(abs_diffs, 0.95, axis=0)
            norms_quantile = jnp.quantile(norms, 0.95)
            return abs_quantile, norms_quantile

        self.risk_sample = jit(risk_sample)

    def __call__(self, batch: jnp.ndarray) -> float:
        self.key, samplekey = random.split(self.key)
        self.mean, self.cov = self._call_helper(self._i, self.mean,
                                                self.cov, batch)
        num_pts = self.batch_size * self._i
        unbiased_cov = num_pts / (num_pts - 1) * self.cov
        feature_err, global_err = self.risk_sample(samplekey,
                                                   unbiased_cov/num_pts)
        self._i += 1
        return feature_err, global_err

    @staticmethod
    @jit
    def _call_helper(i: int, mean: jnp.ndarray, cov: jnp.ndarray,
                     batch: jnp.ndarray) -> jnp.ndarray:
        batch_mean = jnp.mean(batch, axis=0)
        batch_cov = jnp.cov(batch.T, bias=True)

        mean_diff = mean - batch_mean
        correction_term = (i-1) / i**2 * jnp.outer(mean_diff, mean_diff)
        new_mean = (i-1) / i * mean + batch_mean / i
        new_cov = (i-1) / i * cov + batch_cov / i + correction_term
        return new_mean, new_cov

    def reset(self):
        self.mean = jnp.zeros((self.p,))
        self.cov = jnp.zeros((self.p, self.p))
        self._i = 1


class SquareShapley:
    """
    A class for computing the Shapley attribution for a least squares problem with a square data matrix.

    Attributes:
        p (int): The number of features.
        square_shapley (Callable): A function to compute the Shapley
            attribution.
    """

    def __init__(self, p: int):
        self.p = p

    def __call__(self, X_train: jnp.ndarray, X_test: jnp.ndarray,
                 y_train: jnp.ndarray, y_test: jnp.ndarray,
                 y_norm_sq: jnp.ndarray, perms: jnp.ndarray) -> jnp.ndarray:
        return self.square_shapley(X_train, X_test, y_train, y_test,
                                   y_norm_sq, perms)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        self._p = new_p
        def square_shapley(X_train: jnp.ndarray, X_test: jnp.ndarray,
                           y_train: jnp.ndarray, y_test: jnp.ndarray,
                           y_norm_sq: jnp.ndarray,
                           perms: jnp.ndarray) -> jnp.ndarray:
            """
            A helper function to compute the Shapley attribution.

            Args:
                X_train (jnp.ndarray): The training data.
                X_test (jnp.ndarray): The test data.
                y_train (jnp.ndarray): The training labels.
                y_test (jnp.ndarray): The test labels.
                y_norm_sq (jnp.ndarray): The squared norm of the test labels.
                perms (jnp.ndarray): The permutations to use.

            Returns:
                jnp.ndarray: The Shapley attribution.
            """
            
            Q, R = jnp.linalg.qr(X_train[:, perms])
            X = X_test[:, perms]

            Y = jnp.triu(Q.T @ jnp.tile(y_train, (self.p, 1)).T)
            T = jsp.linalg.solve_triangular(R, Y)
            T = jnp.hstack((jnp.zeros((self.p, 1)), T))

            Y_test = jnp.tile(y_test, (self.p+1, 1))
            costs = jnp.sum((X @ T - Y_test.T) ** 2, axis=0)
            R_sq = (jnp.linalg.norm(y_test) ** 2 - costs) / y_norm_sq
            perm_scores = jnp.ediff1d(R_sq)[jnp.argsort(perms)]
            return perm_scores

        vmap_square_shapley = vmap(square_shapley,
                                   (None, None, None, None, None, 0), 0)
        self.square_shapley = jit(vmap_square_shapley)


class LSSPA:
    """
    A class for computing the Shapley attribution for a least squares problem.

    Attributes:
        key (jax.random.PRNGKey): The random number generator key.
        p (int): The number of features.
        sample_method (SampleMethod): The sampling method to use.
        sampler (Permutations): The permutation generator.
        batch_size (int): The batch size.
        _square_shapley (SquareShapley): A function to compute the Shapley
            attribution.
        risk_estimate (RiskEstimate): A function to estimate the risk of the
            Shapley attribution.
    """

    def __init__(self, key, p: int = 10,
                 sample_method: SampleMethod = 'random',
                 batch_size: int = 2**13):
        self._p = p
        self.sample_method = sample_method
        self.key, permkey, riskkey = random.split(key, 3)
        # Initialize appropriate permutation generator based on sampling method
        if self.sample_method == 'random':
            self.sampler = RandomPermutations(permkey, self.p)
        elif self.sample_method == 'permutohedron':
            self.sampler = PermutohedronPermutations(permkey, self.p)
        else:
            self.sampler = ArgsortPermutations(permkey, self.p)

        self.batch_size = batch_size
        self._square_shapley = SquareShapley(p)
        self.risk_estimate = RiskEstimate(riskkey, self.batch_size, self.p)

    def __call__(self, X_train: jnp.ndarray, X_test: jnp.ndarray,
                 y_train: jnp.ndarray, y_test: jnp.ndarray,
                 reg: float, max_num_batches: int = 1,
                 eps: float = 1e-3, y_norm_sq: jnp.ndarray | None = None,
                 return_history: bool = True) -> ShapleyResults:
        if y_norm_sq is None:
            N = 1 if np.isclose(reg, 0) else len(X_train)
            M = len(X_test)
            X_train, X_test, y_train, y_test, y_norm_sq, = (
                self.process_data(N, M, X_train, X_test, y_train, y_test, reg))
        theta = jnp.linalg.lstsq(X_train, y_train)[0]
        # XXX we need to correct the r-squared computation if cholesky method
        # is done. the attributions add up to the right r-squared so maybe we
        # just return that always
        r_squared = (np.linalg.norm(y_test) ** 2 - 
                    np.linalg.norm(X_test @ theta - y_test) ** 2) / y_norm_sq

        attribution_history = jnp.zeros((0, self.p)) if return_history else None
        scores = jnp.zeros(self.p)
        error_history = jnp.zeros((0,)) if return_history else None
        self.risk_estimate.reset()

        for i in range(1, max_num_batches+1):
            batch = self.sampler(self.batch_size)
            perm_scores = self._square_shapley(X_train, X_test, y_train,
                                               y_test, y_norm_sq, batch)
            scores = (i-1)/i * scores + jnp.mean(perm_scores, axis=0) / i
            feature_risk, global_risk = self.risk_estimate(perm_scores)
            if return_history:
                attribution_history = jnp.vstack((attribution_history,
                                                  perm_scores))
                error_history = jnp.append(
                    error_history, global_risk
                )
            if global_risk < eps:
                break

        results = ShapleyResults(attribution=scores,
                                 attribution_history=attribution_history,
                                 theta=theta,
                                 overall_error=global_risk,
                                 error_history=error_history,
                                 attribution_errors=feature_risk,
                                 r_squared=r_squared)
        return results

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, new_p: int):
        if self.p == new_p:
            return

        self.key, permkey, riskkey = random.split(self.key, 3)
        # Initialize appropriate permutation generator based on sampling method
        if self.sample_method == 'random':
            self.sampler = RandomPermutations(permkey, self.p)
        elif self.sample_method == 'permutohedron':
            self.sampler = PermutohedronPermutations(permkey, self.p)
        else:
            self.sampler = ArgsortPermutations(permkey, self.p)

        self._square_shapley = SquareShapley(self.p)
        self.risk_estimate = RiskEstimate(riskkey, self.batch_size, self.p)

    @partial(jit, static_argnums=(0, 1, 2))
    def process_data(self, N: int, M: int, X_train: jnp.ndarray,
                     X_test: jnp.ndarray, y_train: jnp.ndarray,
                     y_test: jnp.ndarray,
                     reg: float) -> Tuple[jnp.ndarray, jnp.ndarray,
                                          jnp.ndarray, jnp.ndarray,
                                          jnp.ndarray]:
        """
        A helper function to reduce the data.

        Args:
            N (int): The number of training observations.
            M (int): The number of test observations.
            X_train (jnp.ndarray): The training data.
            X_test (jnp.ndarray): The test data.
            y_train (jnp.ndarray): The training labels.
            y_test (jnp.ndarray): The test labels.
            reg (float): The regularization parameter.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                  jnp.ndarray]: The reduced data.
        """
        X_train = X_train / jnp.sqrt(N)
        X_train = jnp.vstack((X_train, jnp.sqrt(reg) * jnp.eye(self.p)))
        y_train = y_train / jnp.sqrt(N)
        y_train = jnp.concatenate((y_train, jnp.zeros(self.p)))

        y_norm_sq = jnp.linalg.norm(y_test) ** 2

        Q, X_train, = jnp.linalg.qr(X_train)
        Q_ts, X_test = jnp.linalg.qr(X_test)
        y_train = Q.T @ y_train
        y_test = Q_ts.T @ y_test
        return X_train, X_test, y_train, y_test, y_norm_sq
