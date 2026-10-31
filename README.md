# Gaussian Processes from Scratch: Kernels, Marginal Likelihood, Classification and Scaling

Build Gaussian processes the way chapter 17 of Murphy's Probabilistic Machine Learning teaches them, in numpy with nothing but a Cholesky factorization and an optimizer. Write a kernel library (squared exponential with ARD length scales, Matern, periodic, linear, sums and products), check Mercer's condition on Gram matrices and draw functions from GP priors. Condition on noise-free and noisy data with the stable Cholesky recipe, compute the log marginal likelihood and its data-fit / complexity split, and show that Bayesian linear regression is a GP. Learn kernel hyperparameters from analytic gradients in log space, reproduce the two local optima of the book's Figure 17.9, and fit a composite trend + quasi-periodic + irregular kernel that extrapolates a seasonal series five years ahead where a plain SE kernel falls back to the mean. Then classify iris flowers with a Laplace-approximated GP classifier (SE versus SE + linear), and scale to thousands of points with random Fourier features and inducing points, measuring every approximation against the exact GP by test log-likelihood and by Titsias' bound.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** se_kernel

---

Built on Deep-ML.
