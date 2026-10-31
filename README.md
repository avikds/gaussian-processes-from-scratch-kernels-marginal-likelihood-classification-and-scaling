# Gaussian Processes from Scratch: Kernels, Marginal Likelihood, Classification and Scaling

Build Gaussian processes the way chapter 17 of Murphy's Probabilistic Machine Learning teaches them, in numpy with nothing but a Cholesky factorization and an optimizer. Write a kernel library (squared exponential with ARD length scales, Matern, periodic, linear, sums and products), check Mercer's condition on Gram matrices and draw functions from GP priors. Condition on noise-free and noisy data with the stable Cholesky recipe, compute the log marginal likelihood and its data-fit / complexity split, and show that Bayesian linear regression is a GP. Learn kernel hyperparameters from analytic gradients in log space, reproduce the two local optima of the book's Figure 17.9, and fit a composite trend + quasi-periodic + irregular kernel that extrapolates a seasonal series five years ahead where a plain SE kernel falls back to the mean. Then classify iris flowers with a Laplace-approximated GP classifier (SE versus SE + linear), and scale to thousands of points with random Fourier features and inducing points, measuring every approximation against the exact GP by test log-likelihood and by Titsias' bound.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** se_kernel
- [x] **2.** matern_kernel
- [x] **3.** periodic_kernel
- [x] **4.** linear_kernel
- [x] **5.** kernel_from_spec
- [x] **6.** is_valid_gram
- [x] **7.** sample_gp_prior
- [x] **8.** gp_condition
- [x] **9.** gp_predict
- [x] **10.** log_marginal_likelihood
- [x] **11.** predictive_scores
- [x] **12.** nadaraya_watson
- [x] **13.** bayes_linreg_predict
- [x] **14.** se_kernel_and_grads
- [x] **15.** lml_and_grad
- [x] **16.** maximize_lml
- [x] **17.** lml_surface
- [x] **18.** fit_with_restarts
- [x] **19.** seasonal_kernel_and_grads
- [x] **20.** forecast_comparison
- [x] **21.** laplace_mode
- [x] **22.** gpc_predict
- [x] **23.** iris_gpc_compare

---

Built on Deep-ML.
