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

from functools import partial
import time

import jax.numpy as jnp
import jax.scipy as jsp
from jax import config, devices, device_put, jit, pmap, random, vmap
from jax.lax import cond, fori_loop, scan, while_loop
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import numpy as np
from tqdm import tqdm


def gen_data(rng, conditioning=20, stn_ratio=5):
    # We want to generate a covariance matrix so some features have very large
    # covariances (in magnitude).
    A = rng.standard_normal((p, int(p / conditioning)))
    cov = A @ A.T + np.eye(p)
    v = np.sqrt(np.diag(cov))
    cov = cov / np.outer(v, v)

    # We sample observations to create X_train and X_test.
    X_train = rng.multivariate_normal(np.zeros(p), cov, (N,),
                                      method='svd')
    X_test = rng.multivariate_normal(np.zeros(p), cov, (M,),
                                     method='svd')

    # We want most of the features to be irrelevant.
    theta_vals = np.zeros(p)
    theta_vals[:max((p+1)//10, 1)] = np.full(max((p+1)//10, 1), 2.0)
    theta_true = rng.permutation(theta_vals)

    # We create the response variables and add a little noise.
    std = np.sqrt(np.sum(np.diag(cov) * theta_true**2) / stn_ratio)
    y_train = X_train @ theta_true + std * rng.standard_normal(N)

    X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    X_train = X_train - X_train_mean
    y_train_mean = np.mean(y_train)
    y_train = y_train - y_train_mean

    y_test = X_test @ theta_true + std * rng.standard_normal(M)
    X_test = X_test - X_train_mean
    y_test = y_test - y_train_mean

    return X_train, X_test, y_train, y_test, theta_true, cov


def create_lsspa(p, N, M, K, B, eps, D):
    def reduce_data(X_train, X_test, y_train, y_test, reg):
        X_train = X_train / np.sqrt(N)
        X_train = np.vstack((X_train, np.sqrt(reg) * np.eye(p)))
        y_train = y_train / np.sqrt(N)
        y_train = np.concatenate((y_train, np.zeros(p)))

        sharding = PositionalSharding(mesh_utils.create_device_mesh((D, 1)))
        cat_train = device_put(jnp.hstack((X_train, y_train[:, None])),
                               sharding)
        gram_train = jnp.dot(cat_train.T, cat_train)
        R_train = jsp.linalg.cholesky(gram_train)
        X_train_tilde, y_train_tilde = R_train[:-1, :-1], R_train[:-1, -1]

        cat_test = device_put(jnp.hstack((X_test, y_test[:, None])),
                              sharding)
        gram_test = jnp.dot(cat_test.T, cat_test)
        R_test = jsp.linalg.cholesky(gram_test)
        X_test_tilde, y_test_tilde = R_test[:-1, :-1], R_test[:-1, -1]

        return X_train_tilde, X_test_tilde, y_train_tilde, y_test_tilde


    def single_lift(X_train, X_test, y_train, y_test, y_norm_sq, perm):
        Q, R = jnp.linalg.qr(X_train[:, perm])
        X = X_test[:, perm]

        Y = jnp.triu(Q.T @ jnp.tile(y_train, (p, 1)).T)
        T = jsp.linalg.solve_triangular(R, Y)
        T = jnp.hstack((jnp.zeros((p, 1)), T))

        Y_test = jnp.tile(y_test, (p+1, 1))
        costs = jnp.sum((X @ T - Y_test.T) ** 2, axis=0)
        R_sq = (jnp.sum(y_test ** 2) - costs) / y_norm_sq
        L = jnp.ediff1d(R_sq)[jnp.argsort(perm)]

        return L


    vectorized_lift = vmap(single_lift,
                        in_axes=(None, None, None, None, None, 0),
                        out_axes=0)


    def batched_lift(X_train, X_test, y_train, y_test, y_norm_sq, key):
        key, perm_key = random.split(key)
        to_permute =jnp.tile(jnp.arange(0, p), (B, 1))
        perms = random.permutation(perm_key, to_permute, axis=1,
                                independent=True)

        lifts = vectorized_lift(X_train, X_test, y_train, y_test,
                                y_norm_sq, perms)
        lifts_rev = vectorized_lift(X_train, X_test, y_train, y_test,
                                    y_norm_sq, jnp.flip(perms, 1))
        avg_lifts = (lifts + lifts_rev) / 2
        batch_mean = jnp.mean(avg_lifts, axis=0)
        batch_cov = jnp.cov(avg_lifts.T, bias=True)
        return batch_mean, batch_cov


    parallel_batched_lift = pmap(batched_lift,
                                in_axes=(None, None, None, None, None, 0),
                                out_axes=0)


    def merge_mean_single(old_mean, new_mean, old_N, new_N):
        N = old_N + new_N
        adj_old_mean = (old_N / N) * old_mean
        adj_new_mean = (new_N / N) * new_mean
        return adj_old_mean + adj_new_mean


    def merge_cov_single(old_mean, new_mean, old_cov, new_cov, old_N, new_N):
        N = old_N + new_N
        mean_diff = old_mean - new_mean
        adj_old_cov = (old_N / N) * old_cov
        adj_new_cov = (new_N / N) * new_cov
        delta = (old_N / N) * (new_N / N) * jnp.outer(mean_diff, mean_diff)
        return adj_old_cov + adj_new_cov + delta


    def merge_stats_body(carry, x):
        old_mean, old_cov, old_num = carry
        new_mean = x[0]
        new_cov = x[1:]
        merged_mean = merge_mean_single(old_mean, new_mean, old_num, B)
        merged_cov = merge_cov_single(old_mean, new_mean, old_cov, new_cov,
                                    old_num, B)
        merged_num = old_num + B
        return (merged_mean, merged_cov, merged_num), None


    @jit
    def error_estimate(key, i, cov_b):
        num_samples = (i+1) * B * D
        cov = cov_b / (num_samples - 1)

        key, err_key = random.split(key)
        sample_diffs = random.multivariate_normal(err_key, jnp.zeros(p), cov, shape=(2**12,), method="svd")
        norms = jnp.linalg.norm(sample_diffs, axis=1)
        norms_quantile = jnp.quantile(norms, 0.95)
        return norms_quantile


    @jit
    def update_attrs_cov(i, batch_means, batch_covs, old_mean, old_cov):
        for_scan = jnp.concatenate([jnp.expand_dims(batch_means, 1), batch_covs], axis=1)
        carry, _ = scan(merge_stats_body, (old_mean, old_cov, i*B*D), for_scan)
        new_mean, new_cov, _ = carry
        return new_mean, new_cov


    def lsspa(X_train, X_test, y_train, y_test, y_norm_sq, key):
        attrs = jnp.zeros(p)
        cov = jnp.zeros((p, p))
        err = jnp.inf
        for i in tqdm(range(K // (2 * B * D))):
            if err < eps:
                break
            aux = random.split(key, 1 + D)
            key, perm_keys = aux[0], aux[1:1+D]
            batch_means, batch_covs = parallel_batched_lift(X_train, X_test, y_train,
                                                            y_test, y_norm_sq, perm_keys)
            attrs, cov = update_attrs_cov(i, batch_means, batch_covs, attrs, cov)
            key, err_key = random.split(key)
            err = error_estimate(err_key, i, cov)

        theta = jnp.linalg.lstsq(X_train, y_train, rcond=None)[0]
        r_squared = (jnp.sum(y_test ** 2) - jnp.sum((y_test - X_test @ theta) ** 2)) / y_norm_sq
        return attrs, err, theta, r_squared

    return reduce_data, lsspa


if __name__ == "__main__":
    p = 1000
    N = int(1e6)
    M = int(1e6)
    K = int(2 ** 28)
    B = int(2 ** 8)
    eps = 1e-2
    D = len(devices())
    reduce_data, lsspa = create_lsspa(p, N, M, K, B, eps, D)

    data_rng = np.random.default_rng(42)
    print("Generating data...")
    X_train, X_test, y_train, y_test, true_theta, cov = gen_data(data_rng, conditioning=20, stn_ratio=5)
    print("Data generated.")

    y_norm_sq = np.sum(y_test ** 2)
    red_start = time.time()
    X_train, X_test, y_train, y_test = reduce_data(X_train, X_test, y_train, y_test, 0.0)
    red_end = time.time()

    print(f"Data reduction completed, took {red_end - red_start} seconds.")
    print(f"Condition Number of X_train covariance is {jnp.linalg.cond(X_train)}")
    print(f"Condition Number of X_test covariance is {jnp.linalg.cond(X_test)}")

    key = random.key(0)
    lsspa_start = time.time()
    attrs, err, theta, r_squared = lsspa(X_train, X_test, y_train, y_test,
                                         y_norm_sq, key)
    lsspa_end = time.time()
    print(f"LSSPA completed, took {lsspa_end - lsspa_start} seconds.")
    print(f"Total time: {lsspa_end - lsspa_start + red_end - red_start} seconds.")
    print(f"The estimated error is {err}.")
