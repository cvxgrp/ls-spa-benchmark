import os
import time

import ls_spa
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
from scipy.stats.qmc import MultivariateNormalQMC, Sobol

if not os.path.isdir("./data"):
    os.makedirs("./data")
if not os.path.isdir("./plots"):
    os.makedirs("./plots")

plt.rcdefaults()

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
    })

 
EXP_NAME = "Medium"
max_samples = 2 ** 13
load_gt = True
p = 100
N = int(1e5)
M = int(1e5)
STN_RATIO = 5.
REG = 0.
conditioning = 20.


gt_rng = np.random.default_rng(42)
rng = np.random.default_rng(0)
print("Generating data...")

def gen_data(rng):
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
    std = np.sqrt(np.sum(np.diag(cov) * theta_true**2) / STN_RATIO)
    y_train = X_train @ theta_true + std * rng.standard_normal(N)

    X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    X_train = X_train - X_train_mean
    y_train_mean = np.mean(y_train)
    y_train = y_train - y_train_mean

    y_test = X_test @ theta_true + std * rng.standard_normal(M)
    X_test = X_test - X_train_mean
    y_test = y_test - y_train_mean

    return X_train, X_test, y_train, y_test, theta_true, cov
    
    
def permutohedron_samples(qmc, num_perms: int):
    # Sample on surface of sphere
    samples = qmc.random(num_perms)
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)

    # Project onto permutohedron
    tril_part = np.tril(np.ones((p-1, p)))
    diag_part = np.diag(-np.arange(1, p), 1)[:-1]
    U = tril_part + diag_part
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    samples = samples @ U
    return np.argsort(samples, axis=1)


def argsort_samples(qmc, num_perms: int):
    return np.argsort(qmc.random(num_perms), axis=1)


class GeneratorLen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


class AlternatingGenerator(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length
        self.last_sample = None
        self.next_call_is_direct = True

    def __len__(self):
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            if self.next_call_is_direct:
                self.last_sample = next(self.gen)
                yield self.last_sample
                self.next_call_is_direct = False
            else:
                yield self.last_sample[::-1]
                self.next_call_is_direct = True


X_train, X_test, y_train, y_test, true_theta, cov = gen_data(gt_rng)
print("Data generation complete.")


max_covariance = np.max(np.abs(cov - np.diag(np.diag(cov))))
cond_number = np.linalg.cond(cov)
print(
    f"""The maximum feature covariance is {max_covariance:.2e}, 
    and the condition number of the feature covariance matrix is 
    {cond_number:.2e}."""
)


gt_msg = ""
gt_compute = False
gt_location = f"./data/gt_{EXP_NAME}.npy"
if os.path.exists(gt_location) and load_gt:
    gt_compute, gt_msg = False, "Saved ground-truth attributions loaded."
else:
    gt_compute, gt_msg = True, "Computing ground-truth attributions..."
print(gt_msg)    


gt_attributions = None
if gt_compute:
    gt_permutations_gen = GeneratorLen((gt_rng.permutation(p) for _ in range(2**19)), 2**19)
    gt_permutations = gt_permutations_gen
    gt_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                               perms=gt_permutations, tolerance=0.0)
    gt_attributions = gt_results.attribution
    gt_attributions = gt_attributions * gt_results.r_squared / np.sum(gt_attributions)
    np.save(gt_location, gt_results.attribution)
else:
    gt_attributions = np.load(gt_location)


print("Benchmarking naïve method...")
naive_permutations_gen = GeneratorLen((rng.permutation(p) for _ in range(2**3)), 2**3)
naive_permutations = naive_permutations_gen

def naive_method(batch_size=2**8):
    shapley_values = np.zeros(p)
    attribution_cov = np.zeros((p, p))
    attribution_errors = np.full(p, 0.)
    overall_error = 0.
    error_history = np.zeros(0)
    attribution_history = np.zeros((0, p))

    counter = 0
    for i, perm in enumerate(naive_permutations, 1):
        counter = i
        do_mini_batch = True

        # Compute the lift
        perm = np.array(perm)
        X_train_perm = X_train.copy()[:, perm]
        X_test_perm = X_test.copy()[:, perm]
        lift = np.zeros(p)
        baseline = 0.0
        for j in range(1, p+1):
            theta_fit = np.linalg.lstsq(X_train_perm[:, 0:j], y_train, rcond=None)[0]
            costs = np.sum((X_test_perm[:, 0:j] @ theta_fit - y_test) ** 2)
            R_sq = (np.sum(y_test ** 2) - costs) / np.sum(y_test ** 2)
            lift[perm[j-1]] = R_sq - baseline
            baseline = R_sq

        # Update the mean and biased sample covariance
        attribution_cov = ls_spa.merge_sample_cov(shapley_values, lift,
                                                  attribution_cov, np.zeros((p, p)),
                                                  i-1, 1)
        shapley_values = ls_spa.merge_sample_mean(shapley_values, lift,
                                                  i-1, 1)
        attribution_history = np.vstack((attribution_history, shapley_values))

        # Update the errors
        if (i % batch_size == 0 or i == max_samples - 1):
            unbiased_cov = attribution_cov * i / (i - 1)
            attribution_errors, overall_error = ls_spa.error_estimates(rng,unbiased_cov / i)
            error_history = np.append(error_history, overall_error)
            do_mini_batch = False

    # Last mini-batch
    if do_mini_batch:
        unbiased_cov = attribution_cov * counter / (counter - 1)
        attribution_errors, overall_error = ls_spa.error_estimates(rng, unbiased_cov / i)
        error_history = np.append(error_history, overall_error)

    # Compute auxiliary information
    theta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    r_squared = ((np.linalg.norm(y_test) ** 2
                 - np.linalg.norm(y_test - X_test @ theta) ** 2)
                 / (np.linalg.norm(y_test)**2))

    return ls_spa.ShapleyResults(
        attribution=shapley_values,
        theta=theta,
        overall_error=overall_error,
        error_history=error_history,
        attribution_errors=attribution_errors,
        r_squared=r_squared,
        attribution_history=attribution_history
    )

naive_results = naive_method()
naive_attributions = naive_results.attribution


print("Benchmarking LS-SPA with Monte Carlo sampling, without antithetical sampling...")
mc_samples = (rng.permutation(p) for _ in range(max_samples))
mc_permutations_gen = GeneratorLen(mc_samples, max_samples)
mc_permutations = mc_permutations_gen
mc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                           perms=mc_permutations, tolerance=1e-8,
                           return_attribution_history=True,
                           antithetical=False)
mc_attributions = mc_results.attribution


print("Benchmarking LS-SPA with argsort QMC sampling, without antithetical sampling...")
argsort_qmc = Sobol(p, seed=rng.choice(1000))
argsort_qmc_permutations_gen = GeneratorLen((argsort_samples(argsort_qmc, 1).flatten() for _ in range(max_samples)), max_samples)
argsort_qmc_permutations = argsort_qmc_permutations_gen
argsort_qmc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                    perms=argsort_qmc_permutations, tolerance=1e-8,
                                    return_attribution_history=True,
                                    antithetical=False)
argsort_attributions = argsort_qmc_results.attribution


print("Benchmarking LS-SPA with permutohedron QMC sampling, without antithetical sampling...")
permutohedron_qmc = MultivariateNormalQMC(np.zeros(p-1), seed=rng.choice(1000),
                                          inv_transform=False)
permutohedron_qmc_permutations_gen = GeneratorLen((permutohedron_samples(permutohedron_qmc, 1).flatten() for _ in range(max_samples)), max_samples)
permutohedron_qmc_permutations = permutohedron_qmc_permutations_gen
permutohedron_qmc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                          perms=permutohedron_qmc_permutations,
                                          tolerance=1e-8,
                                          return_attribution_history=True,
                                          antithetical=False)
permutohedron_attributions = permutohedron_qmc_results.attribution


print("Benchmarking LS-SPA with Monte Carlo sampling, with antithetical sampling...")
amc_samples = (rng.permutation(p) for _ in range(max_samples // 2))
amc_permutations_gen = GeneratorLen(amc_samples, max_samples // 2)
amc_permutations = amc_permutations_gen
amc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                            perms=amc_permutations, tolerance=1e-8,
                            return_attribution_history=True,
                            antithetical=True, batch_size=2**7)
amc_attributions = amc_results.attribution


print("Benchmarking LS-SPA with Monte argsort QMC sampling, with antithetical sampling...")
aargsort_qmc = Sobol(p, seed=rng.choice(1000))
aargsort_qmc_permutations_gen = GeneratorLen((argsort_samples(aargsort_qmc, 1).flatten() for _ in range(max_samples//2)), max_samples//2)
aargsort_qmc_permutations = aargsort_qmc_permutations_gen
aargsort_qmc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                    perms=aargsort_qmc_permutations, tolerance=1e-8,
                                    return_attribution_history=True,
                                    antithetical=True)
aargsort_attributions = aargsort_qmc_results.attribution


print("Benchmarking LS-SPA with permutohedron QMC sampling, with antithetical sampling...")
apermutohedron_qmc = MultivariateNormalQMC(np.zeros(p-1), seed=rng.choice(1000),
                                          inv_transform=False)
apermutohedron_qmc_permutations_gen = GeneratorLen((permutohedron_samples(apermutohedron_qmc, 1).flatten() for _ in range(max_samples//2)), max_samples//2)
apermutohedron_qmc_permutations = apermutohedron_qmc_permutations_gen
apermutohedron_qmc_results = ls_spa.ls_spa(X_train, X_test, y_train, y_test,
                                          perms=apermutohedron_qmc_permutations,
                                          tolerance=1e-8,
                                          return_attribution_history=True,
                                          antithetical=True)
apermutohedron_attributions = apermutohedron_qmc_results.attribution


count = np.arange(1, max_samples + 1)
acount = np.arange(1, max_samples + 1, 2)

mc_err = np.linalg.norm(mc_results.attribution_history - gt_attributions, axis=1)
argsort_err = np.linalg.norm(argsort_qmc_results.attribution_history - gt_attributions, axis=1)
permutohedron_err = np.linalg.norm(permutohedron_qmc_results.attribution_history - gt_attributions, axis=1)

amc_err = np.linalg.norm(amc_results.attribution_history - gt_attributions, axis=1)
aargsort_err = np.linalg.norm(aargsort_qmc_results.attribution_history - gt_attributions, axis=1)
apermutohedron_err = np.linalg.norm(apermutohedron_qmc_results.attribution_history - gt_attributions, axis=1)


from matplotlib.lines import Line2D


print("Plotting the true errors against the total number of permutations sampled...")


width_in_inches = 6.3
original_aspect_ratio = 10 / 6  # Original ratio of width to height you had
height_in_inches = width_in_inches / original_aspect_ratio * 1.2

# Specify figsize with the calculated width and height
compare_fig, compare_ax = plt.subplots(figsize=[width_in_inches, height_in_inches])

# Define colors for each method
colors = {
    "MC": "C0",
    "Permutohedron QMC": "C1",
    "Argsort QMC": "C2"
}

# Plot original sampling methods
compare_ax.loglog(count, mc_err, label="MC", color=colors["MC"])
compare_ax.loglog(count, permutohedron_err, label="Permutohedron QMC", color=colors["Permutohedron QMC"])
compare_ax.loglog(count, argsort_err, label="Argsort QMC", color=colors["Argsort QMC"])

# Plot antithetical sampling methods with dashed lines and matching colors
compare_ax.loglog(acount, amc_err, linestyle="dotted", color=colors["MC"])
compare_ax.loglog(acount, apermutohedron_err, linestyle="dotted", color=colors["Permutohedron QMC"])
compare_ax.loglog(acount, aargsort_err, linestyle="dotted", color=colors["Argsort QMC"])

compare_ax.set_xscale("log", base=2)
compare_ax.set_yscale("log", base=10)

# Custom legend
legend_elements = [
    Line2D([0], [0], color=colors["MC"], label='Monte Carlo (MC)'),
    Line2D([0], [0], color=colors["Permutohedron QMC"], label='Permutohedron QMC'),
    Line2D([0], [0], color=colors["Argsort QMC"], label='Argsort QMC'),
    Line2D([0], [0], color='k', linestyle='dotted', label='With Antithetical Samples')
]
# Adjusting the legend's position to the right of the plot
compare_ax.legend(handles=legend_elements, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Adjust layout and spacing to accommodate the legend outside the plot
plt.tight_layout(pad=1.0, h_pad=1.0)
plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin to make room for the legend

plt.xlabel("Total Number of Samples", fontsize=14)
plt.ylabel("Error, $\\|S - \hat S\\|_2$", fontsize=14)
plt.grid(True, which="both", linestyle="--", color="gray",
         linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(f'./plots/err_vs_numsamples_{EXP_NAME}.pdf', format='pdf')


print("Plotting the true and estimated errors for argsort QMC, without antithetical sampling, against the total number of permutations sampled...")


err_fig, err_ax = plt.subplots(figsize=[width_in_inches, height_in_inches])

err_ax.loglog(count[2**8:], argsort_err[2**8:], label="True Error, $\| S -\hat S\|_2$")
err_ax.loglog(np.arange(1, max_samples // (2**8) + 1) * 2**8, argsort_qmc_results.error_history, label=r"Estimated Error, $\hat\sigma$")

err_ax.set_xscale("log", base=2)
err_ax.set_yscale("log", base=10)

plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.xlabel("Number of Samples", fontsize=14)
plt.ylabel("Error", fontsize=14)
plt.grid(True, which="both", linestyle="--", color="gray",
         linewidth=0.5, alpha=0.6)
current_minor_ticks = err_ax.yaxis.get_minor_locator().tick_values(err_ax.get_ylim()[0], err_ax.get_ylim()[1])
updated_minor_ticks = list(current_minor_ticks) + [5e-3]

err_ax.yaxis.set_minor_locator(FixedLocator(updated_minor_ticks))

def custom_formatter(x, pos):
    if x == 5e-3:
        return r'$5\times 10^{-3}$'
    else:
        return ''

err_ax.yaxis.set_minor_formatter(custom_formatter)

plt.tight_layout()
plt.savefig(f'./plots/est_error_argsort_{EXP_NAME}.pdf', format='pdf')



print("Plotting the true and estimated errors for Monte Carlo, with antithetical sampling, against the total number of permutations sampled...")


aerr_fig, aerr_ax = plt.subplots(figsize=[width_in_inches, height_in_inches])

aerr_ax.loglog(count[2**8::2], amc_err[2**7:], label="True Error, $\| S -\hat S\|_2$")
aerr_ax.loglog(np.arange(1, max_samples // (2**8) + 1) * 2**8, amc_results.error_history, label=r"Estimated Error, $\hat\sigma$")

aerr_ax.set_xscale("log", base=2)
aerr_ax.set_yscale("log", base=10)

plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.xlabel("Number of Samples", fontsize=14)
plt.ylabel("Error", fontsize=14)
plt.grid(True, which="both", linestyle="--", color="gray",
         linewidth=0.5, alpha=0.6)
acurrent_minor_ticks = aerr_ax.yaxis.get_minor_locator().tick_values(aerr_ax.get_ylim()[0], aerr_ax.get_ylim()[1])
aupdated_minor_ticks = list(acurrent_minor_ticks) + [5e-3]

aerr_ax.yaxis.set_minor_locator(FixedLocator(aupdated_minor_ticks))
aerr_ax.yaxis.set_minor_formatter(custom_formatter)

plt.tight_layout()
plt.savefig(f'./plots/est_error_antithetical_mc_{EXP_NAME}.pdf', format='pdf')
