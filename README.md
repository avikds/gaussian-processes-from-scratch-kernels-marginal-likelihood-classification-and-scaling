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
- [x] **24.** random_fourier_features
- [x] **25.** rff_gp_predict
- [x] **26.** sparse_gp_predict
- [x] **27.** titsias_bound
- [x] **28.** scaling_showdown

## Results

```
Part 1: kernels and the GP prior
  SE             Mercer check=True  mean |step| of prior draws=0.0432
  Matern 1/2     Mercer check=True  mean |step| of prior draws=0.2559
  Matern 5/2     Mercer check=True  mean |step| of prior draws=0.0505
  periodic       Mercer check=True  mean |step| of prior draws=0.1036
  SE x periodic  Mercer check=True  mean |step| of prior draws=0.0955
  sigmoid 'kernel' tanh(2xx' - 1) passes Mercer: False
Part 2: exact GP regression on the 20 points of Figure 17.8
  (l, sf, sy)=(1.0, 1.0, 0.1): log p(y|X)=-12.085 = fit -13.429 + complexity 19.723 + const -18.379
  (l, sf, sy)=(3.0, 1.16, 0.89): log p(y|X)=-24.382 = fit -3.974 + complexity -2.029 + const -18.379
  weight space vs function space, max |difference| of means: 6.26e-12
Part 3: empirical Bayes
  ML-II on Figure 17.8 data: (l, sf, sy) = [1.022, 1.182, 0.116], log p = -11.594
  Figure 17.9 optimum: l=1.25 sy=0.108 log p=-6.276 (10 of 12 starts)
  Figure 17.9 optimum: l=12.69 sy=0.469 log p=-6.762 (2 of 12 starts)
  marginal-likelihood ratio of the two explanations: 1.63
Part 4: five-year forecast of the seasonal series
  se        train log p= -206.60  test NLL=2.355  RMSE=2.179  95% coverage=0.850
  composite train log p=  -84.60  test NLL=1.321  RMSE=0.868  95% coverage=0.950
  learned period: 0.9995 years
Part 5: GP classification of iris setosa vs versicolor by sepal length (Figure 17.10)
  se        Laplace log p=-43.877  p(versicolor) at [3.0, 4.0, 5.5, 7.0, 8.0]: [0.496, 0.298, 0.567, 0.777, 0.515]
  se+linear Laplace log p=-35.796  p(versicolor) at [3.0, 4.0, 5.5, 7.0, 8.0]: [0.001, 0.005, 0.598, 0.996, 0.999]
Part 6: 2000 training points, M = 200
  hyperparameters (l, sf^2, sy^2): [0.968, 1.782, 0.041]
  exact     test NLL=-0.1322  RMSE=0.2117  95% coverage=0.936
  subset    test NLL=-0.0021  RMSE=0.2413  95% coverage=0.948
  rff       test NLL=0.2868  RMSE=0.2825  95% coverage=0.848
  inducing  test NLL=-0.1318  RMSE=0.2116  95% coverage=0.936
  Titsias bound gap: 0.0026 nats per training point
```
