"""
Gaussian Processes from Scratch: Kernels, Marginal Likelihood, Classification and Scaling scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Gaussian Processes from Scratch (Murphy, Probabilistic Machine Learning: An Introduction, chapter 17).
Kernels and priors (17.1), exact regression and the marginal likelihood (17.2.1-17.2.5), learning the kernel
(17.2.6), a composite seasonal kernel (17.1.2.3), GP classification (17.2.7) and scaling (17.2.9)."""
import numpy as np


def main():
    # --- Part 1: kernels and the GP prior ---
    t = np.linspace(0, 5, 101)
    print("Part 1: kernels and the GP prior")
    for name, K in [("SE", se_kernel(t, t)), ("Matern 1/2", matern_kernel(t, t, 0.5)),
                    ("Matern 5/2", matern_kernel(t, t, 2.5)), ("periodic", periodic_kernel(t, t, 1.0, 2.0)),
                    ("SE x periodic", kernel_from_spec(("*", ("se", {"length_scale": 3.0}), ("periodic", {"period": 2.0})), t, t))]:
        S = sample_gp_prior(K, 3, np.random.default_rng(0))
        print(f"  {name:14s} Mercer check={is_valid_gram(K)[2]}  mean |step| of prior draws={np.abs(np.diff(S, axis=1)).mean():.4f}")
    tt = np.linspace(0, 3, 6)
    print("  sigmoid 'kernel' tanh(2xx' - 1) passes Mercer:", is_valid_gram(np.tanh(2.0 * np.outer(tt, tt) - 1.0))[2])

    # --- Part 2: exact regression ---
    rng = np.random.default_rng(7)
    X = np.sort(rng.uniform(-7, 7, 20))
    y = sample_gp_prior(se_kernel(X, X), 1, rng)[0] + 0.1 * rng.standard_normal(20)
    print("Part 2: exact GP regression on the 20 points of Figure 17.8")
    for ell, sf, sy in [(1.0, 1.0, 0.1), (3.0, 1.16, 0.89)]:
        r = log_marginal_likelihood(X, y, lambda A, B: se_kernel(A, B, ell, sf ** 2), sy ** 2)
        print(f"  (l, sf, sy)=({ell}, {sf}, {sy}): log p(y|X)={r['lml']:.3f} = fit {r['data_fit']:.3f} + complexity {r['complexity']:.3f} + const {r['constant']:.3f}")
    phi = lambda u: np.stack([np.ones_like(np.ravel(u)), np.ravel(u), np.ravel(u) ** 2], 1)
    m1, _ = bayes_linreg_predict(phi(X), y, phi(t), np.eye(3), 0.04)
    m2, _ = gp_predict(X, y, t, lambda A, B: phi(A) @ phi(B).T, 0.04)
    print(f"  weight space vs function space, max |difference| of means: {np.abs(m1 - m2).max():.2e}")

    # --- Part 3: learning the kernel ---
    print("Part 3: empirical Bayes")
    th, lml = maximize_lml(np.log([1.0, 1.0, 0.3]), X, y, se_kernel_and_grads)
    print(f"  ML-II on Figure 17.8 data: (l, sf, sy) = {np.round(np.exp(th), 3).tolist()}, log p = {lml:.3f}")

    def k1(theta, A):
        K, dK = se_kernel_and_grads([theta[0], 0.0], A)
        return K, dK[:1]
    inits = [np.log([l, s]) for l in (0.5, 1, 3, 10) for s in (0.1, 0.3, 1.0)]
    opt = fit_with_restarts(FIG179_X, FIG179_Y, inits, k1)
    for o in opt:
        print(f"  Figure 17.9 optimum: l={np.exp(o['theta'][0]):.2f} sy={np.exp(o['theta'][1]):.3f} log p={o['lml']:.3f} ({o['count']} of {len(inits)} starts)")
    print(f"  marginal-likelihood ratio of the two explanations: {np.exp(opt[0]['lml'] - opt[1]['lml']):.2f}")

    # --- Part 4: composite kernel ---
    print("Part 4: five-year forecast of the seasonal series")
    out = forecast_comparison(*seasonal_series())
    for name in ("se", "composite"):
        r = out[name]
        print(f"  {name:9s} train log p={r['lml']:8.2f}  test NLL={r['nll']:.3f}  RMSE={r['rmse']:.3f}  95% coverage={r['coverage95']:.3f}")
    print(f"  learned period: {np.exp(out['composite']['theta'][5]):.4f} years")

    # --- Part 5: classification ---
    print("Part 5: GP classification of iris setosa vs versicolor by sepal length (Figure 17.10)")
    q = [3.0, 4.0, 5.5, 7.0, 8.0]
    res = iris_gpc_compare(q)
    for name, r in res.items():
        print(f"  {name:9s} Laplace log p={r['lml']:.3f}  p(versicolor) at {q}: {np.round(r['p'], 3).tolist()}")

    # --- Part 6: scaling ---
    print("Part 6: 2000 training points, M = 200")
    r = scaling_showdown()
    print("  hyperparameters (l, sf^2, sy^2):", [round(v, 3) for v in r["hyper"]])
    for name in ("exact", "subset", "rff", "inducing"):
        s = r[name]
        print(f"  {name:9s} test NLL={s['nll']:.4f}  RMSE={s['rmse']:.4f}  95% coverage={s['coverage95']:.3f}")
    print(f"  Titsias bound gap: {r['bound_gap_per_point']:.4f} nats per training point")


if __name__ == "__main__":
    main()

