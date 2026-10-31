"""
Gaussian Processes from Scratch: Kernels, Marginal Likelihood, Classification and Scaling

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - se_kernel
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

def as_2d(X):
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def se_kernel(X1, X2, length_scale=1.0, signal_var=1.0):
    """
    Squared Exponential (RBF) kernel with ARD length scales.

    Parameters
    ----------
    X1 : array-like, shape (n1, D) or (n1,)
        First set of input points.
    X2 : array-like, shape (n2, D) or (n2,)
        Second set of input points.
    length_scale : float or array-like, shape (D,)
        Length scale parameter(s).
    signal_var : float
        Overall signal variance.

    Returns
    -------
    K : ndarray, shape (n1, n2)
        Kernel matrix.
    """

    # Convert inputs to 2D arrays.
    X1 = as_2d(X1)
    X2 = as_2d(X2)

    # Ensure both inputs have the same number of features.
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of features.")

    # Convert length scales to a NumPy array.
    length_scale = np.asarray(length_scale, dtype=float)

    # Validate length scales.
    if np.any(length_scale <= 0):
        raise ValueError("Length scales must be strictly positive.")

    # Validate signal variance.
    if signal_var < 0:
        raise ValueError("Signal variance must be non-negative.")

    # Apply ARD scaling.
    # A scalar length scale is broadcast across all dimensions.
    # A vector provides an individual length scale for each dimension.
    if length_scale.ndim == 0:
        X1_scaled = X1 / length_scale
        X2_scaled = X2 / length_scale

    else:
        if length_scale.ndim != 1 or length_scale.shape[0] != X1.shape[1]:
            raise ValueError(
                "length_scale must be a scalar or an array "
                "with one value per input dimension."
            )

        X1_scaled = X1 / length_scale
        X2_scaled = X2 / length_scale

    # Compute pairwise squared Euclidean distances efficiently.
    # d2[i, j] = ||X1_scaled[i] - X2_scaled[j]||^2

    X1_sq = np.sum(X1_scaled ** 2, axis=1, keepdims=True)
    X2_sq = np.sum(X2_scaled ** 2, axis=1, keepdims=True)

    d2 = X1_sq + X2_sq.T - 2.0 * (X1_scaled @ X2_scaled.T)

    # Clip small negative values caused by floating-point rounding.
    d2 = np.maximum(d2, 0.0)

    # Compute the squared exponential (RBF) kernel.
    K = signal_var * np.exp(-0.5 * d2)

    return K

# Step 2 - matern_kernel
def matern_kernel(X1, X2, nu=1.5, length_scale=1.0, signal_var=1.0):
    """
    Matérn kernel for nu = 0.5, 1.5, or 2.5.

    Parameters
    ----------
    X1 : array-like, shape (n1, D) or (n1,)
        First set of input points.
    X2 : array-like, shape (n2, D) or (n2,)
        Second set of input points.
    nu : float
        Smoothness parameter (0.5, 1.5, or 2.5).
    length_scale : float
        Positive length scale parameter.
    signal_var : float
        Overall signal variance.

    Returns
    -------
    K : ndarray, shape (n1, n2)
        Matérn kernel matrix.
    """

    # Validate the smoothness parameter.
    if nu not in (0.5, 1.5, 2.5):
        raise ValueError("nu must be one of 0.5, 1.5, or 2.5.")

    # Validate the length scale.
    if length_scale <= 0:
        raise ValueError("length_scale must be strictly positive.")

    # Convert inputs to 2D arrays using the helper from Step 1.
    X1 = as_2d(X1)
    X2 = as_2d(X2)

    # Ensure both inputs have the same number of features.
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of features.")

    # Compute pairwise squared Euclidean distances.
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)

    d2 = X1_sq + X2_sq.T - 2.0 * (X1 @ X2.T)

    # Clip negative distances caused by floating-point rounding.
    d2 = np.maximum(d2, 0.0)

    # Compute the Euclidean distance normalized by length_scale.
    r = np.sqrt(d2) / length_scale

    # Matérn kernel for nu = 0.5.
    if nu == 0.5:
        K = signal_var * np.exp(-r)

    # Matérn kernel for nu = 1.5.
    elif nu == 1.5:
        sqrt3_r = np.sqrt(3.0) * r
        K = signal_var * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)

    # Matérn kernel for nu = 2.5.
    else:
        sqrt5_r = np.sqrt(5.0) * r
        K = signal_var * (
            1.0 + sqrt5_r + (5.0 / 3.0) * r**2
        ) * np.exp(-sqrt5_r)

    return K

# Step 3 - periodic_kernel
def periodic_kernel(X1, X2, length_scale=1.0, period=1.0, signal_var=1.0):
    """
    Periodic kernel scaled by a signal variance.

    Parameters
    ----------
    X1 : array-like, shape (n1, D) or (n1,)
        First set of input points.
    X2 : array-like, shape (n2, D) or (n2,)
        Second set of input points.
    length_scale : float
        Positive length scale parameter.
    period : float
        Positive periodicity parameter.
    signal_var : float
        Overall signal variance.

    Returns
    -------
    K : ndarray, shape (n1, n2)
        Periodic kernel matrix.
    """

    # Validate kernel parameters.
    if length_scale <= 0:
        raise ValueError("length_scale must be strictly positive.")

    if period <= 0:
        raise ValueError("period must be strictly positive.")

    if signal_var < 0:
        raise ValueError("signal_var must be non-negative.")

    # Convert inputs into 2D arrays.
    X1 = as_2d(X1)
    X2 = as_2d(X2)

    # Ensure both inputs have the same number of features.
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of features.")

    # Compute pairwise squared Euclidean distances.
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)

    d2 = X1_sq + X2_sq.T - 2.0 * (X1 @ X2.T)

    # Clip small negative values caused by floating-point rounding.
    d2 = np.maximum(d2, 0.0)

    # Compute pairwise Euclidean distances.
    r = np.sqrt(d2)

    # Compute the periodic kernel.
    sin_term = np.sin(np.pi * r / period)

    K = signal_var * np.exp(
        -2.0 * (sin_term ** 2) / (length_scale ** 2)
    )

    return K

# Step 4 - linear_kernel
def linear_kernel(X1, X2, c=0.0, tau=1.0):
    """
    Linear kernel with an offset and scaling parameter.

    Parameters
    ----------
    X1 : array-like, shape (n1, D) or (n1,)
        First set of input points.
    X2 : array-like, shape (n2, D) or (n2,)
        Second set of input points.
    c : float or array-like, shape (D,)
        Offset parameter.
    tau : float
        Kernel scaling parameter.

    Returns
    -------
    K : ndarray, shape (n1, n2)
        Linear kernel matrix.
    """

    # Convert inputs into 2D arrays.
    X1 = as_2d(X1)
    X2 = as_2d(X2)

    # Ensure both inputs have the same number of features.
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of features.")

    # Convert the offset into a NumPy array.
    c = np.asarray(c, dtype=float)

    # Validate the offset dimensions.
    if c.ndim > 1 or (c.ndim == 1 and c.shape[0] != X1.shape[1]):
        raise ValueError(
            "c must be a scalar or an array with one value per feature."
        )

    # Center both input datasets using the offset.
    X1_centered = X1 - c
    X2_centered = X2 - c

    # Compute the scaled linear kernel.
    K = tau * (X1_centered @ X2_centered.T)

    return K

# Step 5 - kernel_from_spec
# Kernel library using functions defined in previous steps.
KERNELS = {
    "se": se_kernel,
    "matern": matern_kernel,
    "periodic": periodic_kernel,
    "linear": linear_kernel
}


def kernel_from_spec(spec, X1, X2):
    """
    Evaluate a kernel defined by a nested specification.

    Parameters
    ----------
    spec : tuple
        Kernel specification.

        Supported formats:
        - ('se',)
        - ('se', {'length_scale': 2.0})
        - ('+', spec1, spec2, ...)
        - ('*', spec1, spec2, ...)

    X1 : array-like
        First set of input points.
    X2 : array-like
        Second set of input points.

    Returns
    -------
    K : ndarray, shape (n1, n2)
        Evaluated kernel matrix.
    """

    # Validate the specification.
    if not isinstance(spec, (tuple, list)) or len(spec) == 0:
        raise ValueError("spec must be a non-empty tuple or list.")

    # Extract the operation or kernel name.
    op = spec[0]

    # Handle composite sum and product kernels.
    if op in ("+", "*"):

        # A composite kernel must contain at least two terms.
        if len(spec) < 3:
            raise ValueError(
                "Sum and product kernels require at least two terms."
            )

        # Evaluate the first child specification recursively.
        K = kernel_from_spec(spec[1], X1, X2)

        # Evaluate the remaining children recursively.
        for child_spec in spec[2:]:

            child_K = kernel_from_spec(child_spec, X1, X2)

            # Apply elementwise addition or multiplication.
            if op == "+":
                K = K + child_K
            else:
                K = K * child_K

        return K

    # Handle leaf kernels.
    else:

        # Check that the leaf specification has a valid structure.
        if len(spec) > 2:
            raise ValueError(
                "A leaf kernel must have the form (name,) or (name, params)."
            )

        # Check whether the requested kernel exists.
        if op not in KERNELS:
            raise ValueError(f"Unknown kernel: {op}")

        # Extract kernel parameters.
        # An omitted parameter dictionary means default parameters.
        params = spec[1] if len(spec) == 2 else {}

        if params is None:
            params = {}

        if not isinstance(params, dict):
            raise ValueError("Kernel parameters must be provided as a dictionary.")

        # Evaluate and return the selected kernel.
        return KERNELS[op](X1, X2, **params)

# Step 6 - is_valid_gram
def is_valid_gram(K, tol=1e-8):
    """
    Check whether a Gram matrix satisfies the Mercer condition.

    Parameters
    ----------
    K : array-like, shape (n, n)
        Gram matrix to be checked.
    tol : float, default=1e-8
        Numerical tolerance for positive semidefiniteness.

    Returns
    -------
    symmetric : bool
        True if K is symmetric within a tolerance of 1e-10.

    min_eig : float
        Smallest eigenvalue of the symmetric part of K.

    valid : bool
        True if K is symmetric and positive semidefinite
        within the specified tolerance.
    """

    # Convert the input into a NumPy array.
    K = np.asarray(K, dtype=float)

    # Ensure that K is a non-empty square matrix.
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] == 0:
        raise ValueError("K must be a non-empty square matrix.")

    # Check symmetry using the specified absolute tolerance.
    symmetric = bool(np.all(np.abs(K - K.T) <= 1e-10))

    # Construct the symmetric part of the Gram matrix.
    K_sym = (K + K.T) / 2.0

    # Compute the eigenvalues of the symmetric matrix.
    # eigvalsh is numerically appropriate for symmetric matrices.
    eigenvalues = np.linalg.eigvalsh(K_sym)

    # Find the smallest eigenvalue.
    min_eig = float(np.min(eigenvalues))

    # Check the Mercer condition within the numerical tolerance.
    valid = bool(symmetric and min_eig >= -tol)

    return symmetric, min_eig, valid

# Step 7 - sample_gp_prior
def sample_gp_prior(K, n_samples, rng, jitter=1e-8):
    """
    Draw function samples from a zero-mean Gaussian Process prior.

    Parameters
    ----------
    K : array-like, shape (n, n)
        Gram matrix (covariance matrix).
    n_samples : int
        Number of functions to sample.
    rng : numpy.random.Generator
        Random number generator.
    jitter : float, default=1e-8
        Small diagonal value added for numerical stability.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Each row represents one sampled function.
    """

    # Convert the Gram matrix to a NumPy array.
    K = np.asarray(K, dtype=float)

    # Get the number of input points.
    n = K.shape[0]

    # Step 1: Compute the lower Cholesky factor of K + jitter * I.
    L = np.linalg.cholesky(K + jitter * np.eye(n))

    # Step 2: Generate standard normal random samples in one call.
    Z = rng.standard_normal((n, n_samples))

    # Step 3: Transform the samples and transpose to obtain
    # an output array of shape (n_samples, n).
    return (L @ Z).T

# Step 8 - gp_condition
def gp_condition(X, f, Xs, kernel, jitter=1e-10):
    """
    Compute the noise-free Gaussian Process posterior.

    Parameters
    ----------
    X : array-like, shape (N, D) or (N,)
        Training inputs.
    f : array-like, shape (N,)
        Observed function values at training inputs.
    Xs : array-like, shape (N_star, D) or (N_star,)
        Test inputs.
    kernel : callable
        Kernel function accepting two input sets and returning
        their Gram matrix.
    jitter : float, default=1e-10
        Small diagonal value added for numerical stability.

    Returns
    -------
    mu : ndarray, shape (N_star,)
        Posterior predictive mean.
    cov : ndarray, shape (N_star, N_star)
        Posterior predictive covariance matrix.
    """

    # Convert training and test inputs into 2D arrays.
    X = as_2d(X)
    Xs = as_2d(Xs)

    # Convert observed function values into a 1D array.
    f = np.asarray(f, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.shape[1] != Xs.shape[1]:
        raise ValueError("X and Xs must have the same number of features.")

    if len(f) != X.shape[0]:
        raise ValueError("The number of observed values must match X.")

    # Number of training points.
    N = X.shape[0]

    # Step 1: Compute the training Gram matrix with numerical jitter.
    K = kernel(X, X) + jitter * np.eye(N)

    # Step 2: Compute the cross-covariance matrix.
    Ks = kernel(X, Xs)

    # Step 3: Compute the test Gram matrix.
    Kss = kernel(Xs, Xs)

    # Step 4: Solve K * alpha = f without explicitly computing K inverse.
    alpha = np.linalg.solve(K, f)

    # Step 5: Compute the posterior predictive mean.
    mu = Ks.T @ alpha

    # Step 6: Solve K * V = Ks.
    V = np.linalg.solve(K, Ks)

    # Step 7: Compute the posterior predictive covariance.
    cov = Kss - Ks.T @ V

    return mu, cov

# Step 9 - gp_predict
def gp_predict(X, y, Xs, kernel, noise_var):
    """
    Compute the noisy-observation Gaussian Process posterior
    using Cholesky factorization.

    Parameters
    ----------
    X : array-like, shape (N, D) or (N,)
        Training inputs.
    y : array-like, shape (N,)
        Noisy observed target values.
    Xs : array-like, shape (N_star, D) or (N_star,)
        Test inputs.
    kernel : callable
        Kernel function returning the Gram matrix.
    noise_var : float
        Observation noise variance.

    Returns
    -------
    mu : ndarray, shape (N_star,)
        Predictive posterior mean.
    var : ndarray, shape (N_star,)
        Predictive latent function variance.
    """

    # Convert inputs into 2D arrays.
    X = as_2d(X)
    Xs = as_2d(Xs)

    # Convert observations into a 1D array.
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if X.shape[1] != Xs.shape[1]:
        raise ValueError("X and Xs must have the same number of features.")

    if len(y) != X.shape[0]:
        raise ValueError("The number of observations must match X.")

    # Validate observation noise variance.
    if noise_var < 0:
        raise ValueError("noise_var must be non-negative.")

    # Number of training observations.
    N = X.shape[0]

    # Step 1: Compute the noisy training covariance matrix.
    K_sigma = kernel(X, X) + noise_var * np.eye(N)

    # Step 2: Compute the lower Cholesky factor.
    L = np.linalg.cholesky(K_sigma)

    # Step 3: Solve L * z = y.
    z = solve_triangular(L, y, lower=True)

    # Step 4: Solve L.T * alpha = z.
    # This gives alpha = K_sigma^(-1) * y without explicitly
    # computing the matrix inverse.
    alpha = solve_triangular(L.T, z, lower=False)

    # Step 5: Compute the cross-covariance matrix.
    Ks = kernel(X, Xs)

    # Step 6: Compute the predictive posterior mean.
    mu = Ks.T @ alpha

    # Step 7: Solve L * v = Ks.
    v = solve_triangular(L, Ks, lower=True)

    # Step 8: Compute the test covariance matrix.
    Kss = kernel(Xs, Xs)

    # Step 9: Compute the predictive latent variance.
    var = np.diag(Kss) - np.sum(v ** 2, axis=0)

    # Clip tiny negative values caused by floating-point rounding.
    var = np.maximum(var, 0.0)

    return mu, var

# Step 10 - log_marginal_likelihood
def log_marginal_likelihood(X, y, kernel, noise_var):
    """
    Compute the Gaussian Process log marginal likelihood
    using Cholesky factorization.

    Parameters
    ----------
    X : array-like, shape (N, D) or (N,)
        Training inputs.
    y : array-like, shape (N,)
        Observed target values.
    kernel : callable
        Kernel function returning the Gram matrix.
    noise_var : float
        Observation noise variance.

    Returns
    -------
    result : dict
        Dictionary containing:
        - data_fit
        - complexity
        - constant
        - lml
    """

    # Convert inputs into 2D arrays.
    X = as_2d(X)

    # Convert observations into a 1D array.
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate input dimensions.
    if len(y) != X.shape[0]:
        raise ValueError("The number of observations must match X.")

    if noise_var < 0:
        raise ValueError("noise_var must be non-negative.")

    # Number of observations.
    N = X.shape[0]

    # Step 1: Construct the noisy covariance matrix.
    K_sigma = kernel(X, X) + noise_var * np.eye(N)

    # Step 2: Compute the lower Cholesky factor.
    L = np.linalg.cholesky(K_sigma)

    # Step 3: Compute alpha = K_sigma^(-1) @ y
    # using Cholesky-based linear system solving.
    alpha = cho_solve((L, True), y)

    # Step 4: Calculate the data-fit term.
    data_fit = -0.5 * float(y @ alpha)

    # Step 5: Calculate the model-complexity term.
    complexity = -float(np.sum(np.log(np.diag(L))))

    # Step 6: Calculate the Gaussian normalization constant.
    constant = -0.5 * N * np.log(2.0 * np.pi)

    # Step 7: Calculate the total log marginal likelihood.
    lml = data_fit + complexity + constant

    # Return all terms as Python floats.
    return {
        "data_fit": float(data_fit),
        "complexity": float(complexity),
        "constant": float(constant),
        "lml": float(lml)
    }

# Step 11 - predictive_scores
def predictive_scores(y, mu, var):
    """
    Evaluate Gaussian predictive distributions against observed targets.

    Parameters
    ----------
    y : array-like, shape (N,)
        Actual observed target values.
    mu : array-like, shape (N,)
        Predictive means.
    var : array-like, shape (N,)
        Predictive variances of the noisy targets.

    Returns
    -------
    scores : dict
        Dictionary containing:
        - nll: Mean negative log predictive density.
        - rmse: Root mean squared error.
        - coverage95: Fraction of observations within the 95%
          predictive intervals.
    """

    # Convert inputs into NumPy arrays.
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    var = np.asarray(var, dtype=float).reshape(-1)

    # Ensure all three arrays have the same length.
    if not (len(y) == len(mu) == len(var)):
        raise ValueError("y, mu, and var must have the same length.")

    # Ensure the arrays are not empty.
    if len(y) == 0:
        raise ValueError("Input arrays must not be empty.")

    # Predictive variances must be strictly positive.
    if np.any(var <= 0):
        raise ValueError("Predictive variances must be strictly positive.")

    # Compute prediction errors.
    errors = y - mu

    # 1. Mean Gaussian Negative Log-Likelihood (NLL).
    nll = np.mean(
        0.5 * np.log(2.0 * np.pi * var)
        + (errors ** 2) / (2.0 * var)
    )

    # 2. Root Mean Squared Error (RMSE).
    rmse = np.sqrt(np.mean(errors ** 2))

    # 3. 95% predictive interval coverage.
    # An observation is covered if it falls within:
    # mu +/- 1.96 * sqrt(var).
    covered = np.abs(errors) <= 1.96 * np.sqrt(var)

    coverage95 = np.mean(covered)

    # Return all metrics as Python floats.
    return {
        "nll": float(nll),
        "rmse": float(rmse),
        "coverage95": float(coverage95)
    }

# Step 12 - nadaraya_watson
def nadaraya_watson(X, y, Xs, bandwidth):
    """
    Perform Nadaraya-Watson kernel regression using a Gaussian kernel.

    Parameters
    ----------
    X : array-like, shape (N, D) or (N,)
        Training inputs.
    y : array-like, shape (N,)
        Observed target values.
    Xs : array-like, shape (N_star, D) or (N_star,)
        Test inputs.
    bandwidth : float
        Positive bandwidth parameter of the Gaussian kernel.

    Returns
    -------
    predictions : ndarray, shape (N_star,)
        Predicted values at the test inputs.
    """

    # Convert inputs into 2D arrays.
    X = as_2d(X)
    Xs = as_2d(Xs)

    # Convert target values into a 1D array.
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate the bandwidth.
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be strictly positive.")

    # Validate input dimensions.
    if X.shape[1] != Xs.shape[1]:
        raise ValueError("X and Xs must have the same number of features.")

    # Validate the number of training targets.
    if len(y) != X.shape[0]:
        raise ValueError("The number of target values must match X.")

    if X.shape[0] == 0:
        raise ValueError("At least one training point is required.")

    # Compute pairwise squared Euclidean distances.
    Xs_sq = np.sum(Xs ** 2, axis=1, keepdims=True)
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)

    d2 = Xs_sq + X_sq.T - 2.0 * (Xs @ X.T)

    # Clip small negative values caused by floating-point rounding.
    d2 = np.maximum(d2, 0.0)

    # Compute Gaussian log weights.
    log_weights = -d2 / (2.0 * bandwidth ** 2)

    # Numerical stabilization:
    # Subtract the maximum log weight in each row before exponentiation.
    log_weights -= np.max(log_weights, axis=1, keepdims=True)

    # Convert log weights into Gaussian weights.
    weights = np.exp(log_weights)

    # Normalize weights separately for each test input.
    weights /= np.sum(weights, axis=1, keepdims=True)

    # Compute predictions using normalized weighted averages.
    predictions = weights @ y

    return predictions

# Step 13 - bayes_linreg_predict
def bayes_linreg_predict(Phi, y, Phi_star, Sigma_w, noise_var):
    """
    Compute the weight-space posterior predictive distribution
    of Bayesian Linear Regression.

    Parameters
    ----------
    Phi : array-like, shape (N, D)
        Training design matrix.
    y : array-like, shape (N,)
        Observed target values.
    Phi_star : array-like, shape (N_star, D)
        Test design matrix.
    Sigma_w : array-like, shape (D, D)
        Prior covariance matrix of the weights.
    noise_var : float
        Observation noise variance.

    Returns
    -------
    mean : ndarray, shape (N_star,)
        Posterior predictive mean.
    var : ndarray, shape (N_star,)
        Posterior predictive latent variance.
    """

    # Convert input matrices into NumPy arrays.
    Phi = as_2d(Phi)
    Phi_star = as_2d(Phi_star)

    y = np.asarray(y, dtype=float).reshape(-1)
    Sigma_w = np.asarray(Sigma_w, dtype=float)

    # Validate dimensions.
    N, D = Phi.shape

    if len(y) != N:
        raise ValueError("The number of target values must match Phi.")

    if Phi_star.shape[1] != D:
        raise ValueError("Phi and Phi_star must have the same number of features.")

    if Sigma_w.shape != (D, D):
        raise ValueError("Sigma_w must have shape (D, D).")

    if not np.all(np.isfinite(Phi)) or not np.all(np.isfinite(Phi_star)):
        raise ValueError("Design matrices must contain finite values.")

    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(Sigma_w)):
        raise ValueError("y and Sigma_w must contain finite values.")

    if not np.isfinite(noise_var) or noise_var <= 0:
        raise ValueError("noise_var must be strictly positive.")

    # Ensure the prior covariance matrix is symmetric.
    if not np.allclose(Sigma_w, Sigma_w.T, atol=1e-10, rtol=0.0):
        raise ValueError("Sigma_w must be symmetric.")

    # Step 1: Compute the inverse of the prior covariance matrix.
    # This involves only a D x D matrix.
    Sigma_w_inv = np.linalg.solve(Sigma_w, np.eye(D))

    # Step 2: Compute the posterior precision matrix A.
    A = (Phi.T @ Phi) / noise_var + Sigma_w_inv

    # Remove tiny numerical asymmetries.
    A = 0.5 * (A + A.T)

    # Step 3: Compute the lower Cholesky factor of A.
    L = np.linalg.cholesky(A)

    # Step 4: Compute the right-hand side for the posterior weights.
    b = (Phi.T @ y) / noise_var

    # Step 5: Solve A * w_mean = b using Cholesky factorization.
    w_mean = cho_solve((L, True), b)

    # Step 6: Compute the posterior predictive mean.
    mean = Phi_star @ w_mean

    # Step 7: Solve L * V = Phi_star.T.
    V = solve_triangular(
        L,
        Phi_star.T,
        lower=True
    )

    # Step 8: Compute the posterior predictive latent variance.
    var = np.sum(V ** 2, axis=0)

    return mean, var

# Step 14 - se_kernel_and_grads
def se_kernel_and_grads(log_params, X):
    """
    Compute the Squared Exponential kernel matrix and its
    analytical gradients with respect to log hyperparameters.

    Parameters
    ----------
    log_params : array-like, shape (2,)
        Log hyperparameters:
        [log_ell, log_sigma_f].

    X : array-like, shape (N, D) or (N,)
        Training input points.

    Returns
    -------
    K : ndarray, shape (N, N)
        Squared Exponential kernel matrix.

    gradients : list
        Two gradient matrices:
        - dK_dlog_ell
        - dK_dlog_sigma_f
    """

    # Convert inputs into a 2D array.
    X = as_2d(X)

    # Extract the log hyperparameters.
    log_ell, log_sigma_f = np.asarray(log_params, dtype=float)

    # Convert log hyperparameters into positive parameters.
    ell = np.exp(log_ell)
    sigma_f = np.exp(log_sigma_f)

    # Signal variance.
    sf2 = sigma_f ** 2

    # Compute squared Euclidean distances between all input pairs.
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)

    r2 = X_sq + X_sq.T - 2.0 * (X @ X.T)

    # Clip small negative distances caused by floating-point rounding.
    r2 = np.maximum(r2, 0.0)

    # Compute the Squared Exponential kernel matrix.
    K = sf2 * np.exp(-r2 / (2.0 * ell ** 2))

    # Gradient with respect to log length scale.
    dK_dlog_ell = K * r2 / (ell ** 2)

    # Gradient with respect to log signal standard deviation.
    dK_dlog_sigma_f = 2.0 * K

    # Return the kernel matrix and its gradient matrices.
    return K, [dK_dlog_ell, dK_dlog_sigma_f]

# Step 15 - lml_and_grad
def lml_and_grad(theta, X, y, kernel_and_grads):
    """
    Compute the GP log marginal likelihood and its analytical
    gradients with respect to log hyperparameters.

    Parameters
    ----------
    theta : array-like, shape (M + 1,)
        Log hyperparameters. The last element is the log noise
        standard deviation; the remaining elements are kernel
        log hyperparameters.
    X : array-like
        Training inputs.
    y : array-like, shape (N,)
        Observed target values.
    kernel_and_grads : callable
        Function returning the kernel matrix and its derivatives:
        K, [dK_1, dK_2, ..., dK_M].

    Returns
    -------
    lml : float
        Log marginal likelihood.

    grad : ndarray, shape (M + 1,)
        Analytical gradient with respect to log hyperparameters.
    """

    # Convert inputs into NumPy arrays.
    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate hyperparameters.
    if theta.ndim != 1 or len(theta) < 1:
        raise ValueError("theta must be a non-empty 1D array.")

    if not np.all(np.isfinite(theta)):
        raise ValueError("theta must contain finite values.")

    # Number of training observations.
    N = len(y)

    if N == 0:
        raise ValueError("At least one training observation is required.")

    # Step 1: Compute the kernel matrix and its derivatives.
    K, dKs = kernel_and_grads(theta[:-1], X)

    K = np.asarray(K, dtype=float)

    if K.shape != (N, N):
        raise ValueError("The kernel matrix must have shape (N, N).")

    if len(dKs) != len(theta) - 1:
        raise ValueError("The number of kernel gradients must match theta.")

    # Step 2: Convert the log noise standard deviation
    # into the observation noise variance.
    noise_var = np.exp(2.0 * theta[-1])

    # Step 3: Construct the noisy covariance matrix.
    K_sigma = K + noise_var * np.eye(N)

    # Step 4: Compute the lower Cholesky factorization.
    L = np.linalg.cholesky(K_sigma)

    # Step 5: Compute alpha = K_sigma^(-1) @ y
    # using Cholesky-based linear system solving.
    alpha = cho_solve((L, True), y)

    # Step 6: Compute the log marginal likelihood.
    # The Cholesky diagonal gives the log determinant term.
    lml = (
        -0.5 * (y @ alpha)
        - np.sum(np.log(np.diag(L)))
        - 0.5 * N * np.log(2.0 * np.pi)
    )

    # Step 7: Compute the inverse covariance matrix
    # using Cholesky-based solving, without np.linalg.inv().
    K_inv = cho_solve((L, True), np.eye(N))

    # Step 8: Compute W = alpha @ alpha.T - K_sigma^(-1).
    W = np.outer(alpha, alpha) - K_inv

    # Step 9: Compute analytical gradients for kernel
    # hyperparameters.
    grad = np.empty(len(theta), dtype=float)

    for j, dK in enumerate(dKs):
        dK = np.asarray(dK, dtype=float)

        if dK.shape != (N, N):
            raise ValueError("Every kernel gradient must have shape (N, N).")

        grad[j] = 0.5 * np.sum(W * dK)

    # Step 10: Compute the gradient with respect to
    # log noise standard deviation.
    grad[-1] = noise_var * np.trace(W)

    return float(lml), grad

# Step 16 - maximize_lml
def maximize_lml(theta0, X, y, kernel_and_grads, bound=5.0, max_iter=500):
    """
    Maximize the Gaussian Process log marginal likelihood
    using L-BFGS-B optimization.

    Parameters
    ----------
    theta0 : array-like
        Initial log hyperparameters.
        The last element represents log noise standard deviation.

    X : array-like
        Training inputs.

    y : array-like
        Observed target values.

    kernel_and_grads : callable
        Function returning the kernel matrix and its analytical
        gradients.

    bound : float, default=5.0
        Absolute bound applied to each log hyperparameter.

    max_iter : int, default=500
        Maximum number of optimization iterations.

    Returns
    -------
    theta : ndarray
        Optimized log hyperparameters.

    lml : float
        Maximized log marginal likelihood.
    """

    # Convert initial hyperparameters into a NumPy array.
    theta0 = np.asarray(theta0, dtype=float)

    if theta0.ndim != 1 or theta0.size == 0:
        raise ValueError("theta0 must be a non-empty 1D array.")

    if not np.all(np.isfinite(theta0)):
        raise ValueError("theta0 must contain finite values.")

    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be strictly positive.")

    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    # Objective function for minimization.
    # We minimize the negative log marginal likelihood.
    def objective(theta):
        try:
            # Compute the log marginal likelihood and its gradient.
            lml, grad = lml_and_grad(
                theta,
                X,
                y,
                kernel_and_grads
            )

            # Convert the maximization problem into minimization.
            loss = -lml
            grad = -np.asarray(grad, dtype=float)

            # Guard against invalid numerical results.
            if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
                return 1e10, np.zeros_like(theta)

            return float(loss), grad

        except np.linalg.LinAlgError:
            # If Cholesky factorization fails, return a large penalty
            # and a zero gradient so the optimizer can backtrack.
            return 1e10, np.zeros_like(theta)

    # Apply identical bounds to every log hyperparameter.
    bounds = [(-bound, bound)] * len(theta0)

    # Run L-BFGS-B optimization with analytical gradients.
    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter}
    )

    # Return optimized hyperparameters and maximum LML.
    return res.x, float(-res.fun)

# Step 17 - lml_surface
# Seven observations from Figure 17.9.
FIG179_X = [-4.99, -3.82, 0.44, 1.04, 1.26, 2.02, 3.21]
FIG179_Y = [-0.28, -0.77, -0.89, -0.64, -0.47, -1.07, -1.94]


def lml_surface(X, y, length_scales, noise_stds, signal_var=1.0):
    """
    Compute the Gaussian Process log marginal likelihood surface
    for different length scales and observation noise levels.

    Identify all interior local maxima that are strictly greater
    than their eight neighbouring grid points.

    Parameters
    ----------
    X : array-like
        Training inputs.
    y : array-like
        Observed target values.
    length_scales : array-like
        Length scale values (columns).
    noise_stds : array-like
        Noise standard deviation values (rows).
    signal_var : float
        Fixed signal variance.

    Returns
    -------
    G : ndarray
        Log marginal likelihood surface.
    maxima : list
        Interior local maxima as (i, j) tuples in row-major order.
    """

    # Convert parameter grids to NumPy arrays.
    length_scales = np.asarray(length_scales, dtype=float).reshape(-1)
    noise_stds = np.asarray(noise_stds, dtype=float).reshape(-1)

    # Validate inputs.
    if length_scales.size == 0 or noise_stds.size == 0:
        raise ValueError("Parameter grids must not be empty.")

    if np.any(~np.isfinite(length_scales)) or np.any(length_scales <= 0):
        raise ValueError("Length scales must be finite and positive.")

    if np.any(~np.isfinite(noise_stds)) or np.any(noise_stds < 0):
        raise ValueError("Noise standard deviations must be finite and non-negative.")

    if not np.isfinite(signal_var) or signal_var < 0:
        raise ValueError("Signal variance must be finite and non-negative.")

    # Initialize the LML surface.
    # Rows correspond to noise levels.
    # Columns correspond to length scales.
    G = np.empty(
        (len(noise_stds), len(length_scales)),
        dtype=float
    )

    # Evaluate the log marginal likelihood at every grid point.
    for i, noise_std in enumerate(noise_stds):

        # Convert noise standard deviation into noise variance.
        noise_var = noise_std ** 2

        for j, length_scale in enumerate(length_scales):

            # Define the squared exponential kernel.
            # The default argument preserves the current length scale.
            kernel = lambda A, B, ell=length_scale: se_kernel(
                A,
                B,
                length_scale=ell,
                signal_var=signal_var
            )

            # Calculate the log marginal likelihood.
            result = log_marginal_likelihood(
                X,
                y,
                kernel,
                noise_var
            )

            G[i, j] = result["lml"]

    # Find interior local maxima.
    maxima = []

    for i in range(1, G.shape[0] - 1):
        for j in range(1, G.shape[1] - 1):

            # Extract the 3 x 3 neighbourhood.
            neighbourhood = G[i-1:i+2, j-1:j+2].copy()

            # Exclude the centre cell.
            neighbourhood[1, 1] = -np.inf

            # Check whether the centre is strictly greater
            # than all eight neighbouring cells.
            if np.all(G[i, j] > neighbourhood):
                maxima.append((int(i), int(j)))

    return G, maxima

# Step 18 - fit_with_restarts
def fit_with_restarts(X, y, inits, kernel_and_grads, tol=0.1):
    """
    Fit Gaussian Process hyperparameters using multiple restarts.

    Parameters
    ----------
    X : array-like
        Training inputs.
    y : array-like
        Observed target values.
    inits : iterable
        Collection of initial log-hyperparameter arrays.
    kernel_and_grads : callable
        Function returning the kernel matrix and its analytical
        gradients.
    tol : float, default=0.1
        Maximum absolute difference in log-parameter space
        for considering two solutions identical.

    Returns
    -------
    optima : list of dictionaries
        Distinct optimized solutions, sorted by descending LML.

        Each dictionary contains:
        - theta: optimized log-hyperparameter array
        - lml: maximum log marginal likelihood
        - count: number of initializations converging to this optimum
    """

    # Validate tolerance.
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite, non-negative value.")

    # Store all distinct local optima.
    optima = []

    # Process every initialization in its original order.
    for theta0 in inits:

        # Optimize the log marginal likelihood from this initialization.
        theta_opt, lml_opt = maximize_lml(
            theta0,
            X,
            y,
            kernel_and_grads
        )

        theta_opt = np.asarray(theta_opt, dtype=float)
        lml_opt = float(lml_opt)

        # Search for an existing optimum within the specified tolerance.
        matched_optimum = None

        for optimum in optima:

            # Compare optimized parameters in log space.
            difference = np.max(
                np.abs(optimum["theta"] - theta_opt)
            )

            if difference < tol:
                matched_optimum = optimum
                break

        # If an existing optimum is found, update its count.
        if matched_optimum is not None:

            matched_optimum["count"] += 1

            # Keep the result with the higher log marginal likelihood.
            if lml_opt > matched_optimum["lml"]:
                matched_optimum["theta"] = theta_opt.copy()
                matched_optimum["lml"] = lml_opt

        # Otherwise, create a new distinct optimum.
        else:

            optima.append({
                "theta": theta_opt.copy(),
                "lml": lml_opt,
                "count": 1
            })

    # Sort distinct optima by log marginal likelihood,
    # with the highest value first.
    optima.sort(
        key=lambda optimum: optimum["lml"],
        reverse=True
    )

    return optima

# Step 19 - seasonal_kernel_and_grads
# ---------------------------------------------------------
# Given seasonal time-series data generator.
# ---------------------------------------------------------

def seasonal_series(n_years=20, per_year=8, noise_std=0.3, seed=0):
    rng = np.random.default_rng(seed)

    t = np.arange(n_years * per_year) / per_year

    # Long-term trend.
    trend = 0.6 * t + 0.015 * t ** 2

    # Slowly varying seasonal amplitude.
    amp = 1.0 + 0.25 * np.sin(2 * np.pi * t / 13.0)

    # Yearly and half-yearly seasonal components.
    season = amp * (
        1.5 * np.sin(2 * np.pi * t)
        + 0.5 * np.sin(4 * np.pi * t + 0.7)
    )

    # Irregular oscillations.
    irregular = (
        0.6 * np.sin(2 * np.pi * t / 3.3 + 1.0)
        + 0.4 * np.sin(2 * np.pi * t / 1.7 + 2.0)
    )

    # Add Gaussian observation noise.
    y = trend + season + irregular + noise_std * rng.standard_normal(len(t))

    return t, y


# ---------------------------------------------------------
# Given composite kernel specification.
# ---------------------------------------------------------

def seasonal_spec(log_params):
    s1, l1, s2, l2, l3, p, s3, l4, tau = np.exp(
        np.asarray(log_params, dtype=float)
    )

    return (
        "+",
        ("se", {"length_scale": l1, "signal_var": s1}),
        (
            "*",
            ("se", {"length_scale": l2, "signal_var": s2}),
            ("periodic", {"length_scale": l3, "period": p})
        ),
        ("se", {"length_scale": l4, "signal_var": s3}),
        ("linear", {"c": 0.0, "tau": tau})
    )


# ---------------------------------------------------------
# Seasonal composite kernel and analytical gradients.
# ---------------------------------------------------------

def seasonal_kernel_and_grads(log_params, X):
    """
    Compute the four-component seasonal composite kernel
    and its nine analytical log-parameter derivatives.

    Parameter order:
    [s1, l1, s2, l2, l3, p, s3, l4, tau]

    Returns
    -------
    K : ndarray, shape (N, N)
        Composite kernel matrix.

    dKs : list
        Nine derivative matrices in parameter order.
    """

    # Convert inputs to NumPy arrays.
    log_params = np.asarray(log_params, dtype=float)
    X = as_2d(X)

    # Validate input parameters.
    if log_params.shape != (9,):
        raise ValueError("log_params must contain exactly nine values.")

    if not np.all(np.isfinite(log_params)):
        raise ValueError("log_params must contain finite values.")

    # Transform log parameters into positive kernel parameters.
    s1, l1, s2, l2, l3, p, s3, l4, tau = np.exp(log_params)

    # Compute pairwise squared Euclidean distances.
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)

    r2 = X_sq + X_sq.T - 2.0 * (X @ X.T)

    # Prevent small negative squared distances from rounding.
    r2 = np.maximum(r2, 0.0)

    # Pairwise Euclidean distances.
    r = np.sqrt(r2)

    # -----------------------------------------------------
    # Component 1: Long-term squared exponential trend.
    # K1 = s1 * SE(l1)
    # -----------------------------------------------------

    K1 = s1 * np.exp(
        -r2 / (2.0 * l1 ** 2)
    )

    # -----------------------------------------------------
    # Component 2: Quasi-periodic seasonal component.
    # K2 = s2 * SE(l2) * Periodic(l3, p)
    # -----------------------------------------------------

    SE2 = np.exp(
        -r2 / (2.0 * l2 ** 2)
    )

    u = np.pi * r / p

    periodic = np.exp(
        -2.0 * np.sin(u) ** 2 / (l3 ** 2)
    )

    K2 = s2 * SE2 * periodic

    # -----------------------------------------------------
    # Component 3: Short-term irregular component.
    # K3 = s3 * SE(l4)
    # -----------------------------------------------------

    K3 = s3 * np.exp(
        -r2 / (2.0 * l4 ** 2)
    )

    # -----------------------------------------------------
    # Component 4: Linear trend.
    # K4 = tau * X @ X.T
    # -----------------------------------------------------

    K4 = tau * (X @ X.T)

    # -----------------------------------------------------
    # Construct the complete kernel using the given spec.
    # -----------------------------------------------------

    K = kernel_from_spec(
        seasonal_spec(log_params),
        X,
        X
    )

    # -----------------------------------------------------
    # Analytical derivatives with respect to log parameters.
    # -----------------------------------------------------

    # 1. Derivative with respect to log(s1).
    dK_dlog_s1 = K1

    # 2. Derivative with respect to log(l1).
    dK_dlog_l1 = K1 * r2 / (l1 ** 2)

    # 3. Derivative with respect to log(s2).
    dK_dlog_s2 = K2

    # 4. Derivative with respect to log(l2).
    dK_dlog_l2 = K2 * r2 / (l2 ** 2)

    # 5. Derivative with respect to log(l3).
    dK_dlog_l3 = (
        K2 * 4.0 * np.sin(u) ** 2 / (l3 ** 2)
    )

    # 6. Derivative with respect to log(period).
    dK_dlog_p = (
        K2 * 4.0 * u * np.sin(u) * np.cos(u) / (l3 ** 2)
    )

    # 7. Derivative with respect to log(s3).
    dK_dlog_s3 = K3

    # 8. Derivative with respect to log(l4).
    dK_dlog_l4 = K3 * r2 / (l4 ** 2)

    # 9. Derivative with respect to log(tau).
    dK_dlog_tau = K4

    # Return derivatives in exactly the specified order.
    dKs = [
        dK_dlog_s1,
        dK_dlog_l1,
        dK_dlog_s2,
        dK_dlog_l2,
        dK_dlog_l3,
        dK_dlog_p,
        dK_dlog_s3,
        dK_dlog_l4,
        dK_dlog_tau
    ]

    return K, dKs

# Step 20 - forecast_comparison
def forecast_comparison(t, y, split_year=15.0):
    """
    Compare a plain SE Gaussian Process with a composite
    seasonal Gaussian Process for future forecasting.

    Parameters
    ----------
    t : array-like, shape (N,)
        Time points in years.
    y : array-like, shape (N,)
        Observed time-series values.
    split_year : float, default=15.0
        Training observations satisfy t < split_year.
        Test observations satisfy t >= split_year.

    Returns
    -------
    results : dict
        Dictionary containing results for both models:
        - se
        - composite

        Each model contains:
        - nll
        - rmse
        - coverage95
        - lml
        - theta
        - mean
    """

    # -----------------------------------------------------
    # Step 1: Prepare and split the dataset.
    # -----------------------------------------------------

    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(t) != len(y):
        raise ValueError("t and y must have the same length.")

    if len(t) == 0:
        raise ValueError("The input dataset must not be empty.")

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("t and y must contain finite values.")

    train_mask = t < split_year
    test_mask = t >= split_year

    t_train = t[train_mask]
    y_train = y[train_mask]

    t_test = t[test_mask]
    y_test = y[test_mask]

    if len(t_train) == 0 or len(t_test) == 0:
        raise ValueError("Both training and test sets must be non-empty.")

    # Compute the training mean and centre all targets.
    y_bar = np.mean(y_train)

    y_train_centered = y_train - y_bar
    y_test_centered = y_test - y_bar

    # Standard deviation of the centred training targets.
    s = np.std(y_train_centered)

    if s <= 0:
        raise ValueError("Training targets must have positive standard deviation.")

    # -----------------------------------------------------
    # Step 2: Fit the plain Squared Exponential GP.
    # -----------------------------------------------------

    se_theta0 = np.log([5.0, s, 0.5])

    se_theta, se_lml = maximize_lml(
        se_theta0,
        t_train,
        y_train_centered,
        se_kernel_and_grads
    )

    # Extract fitted SE hyperparameters.
    se_length_scale = np.exp(se_theta[0])
    se_signal_var = np.exp(2.0 * se_theta[1])
    se_noise_var = np.exp(2.0 * se_theta[2])

    # Construct the fitted SE kernel.
    def fitted_se_kernel(A, B):
        return se_kernel(
            A,
            B,
            length_scale=se_length_scale,
            signal_var=se_signal_var
        )

    # Predict future observations.
    se_mu, se_var = gp_predict(
        t_train,
        y_train_centered,
        t_test,
        fitted_se_kernel,
        se_noise_var
    )

    # Score predictions using latent variance plus observation noise.
    se_scores = predictive_scores(
        y_test_centered,
        se_mu,
        se_var + se_noise_var
    )

    # -----------------------------------------------------
    # Step 3: Fit the composite seasonal GP.
    # -----------------------------------------------------

    composite_theta0 = np.log([
        10.0, 20.0, 2.0, 30.0, 1.0,
        1.0, 0.5, 1.0, 0.01, 0.3
    ])

    composite_theta, composite_lml = maximize_lml(
        composite_theta0,
        t_train,
        y_train_centered,
        seasonal_kernel_and_grads
    )

    # Extract fitted observation noise variance.
    composite_noise_var = np.exp(2.0 * composite_theta[-1])

    # Construct the fitted composite kernel.
    def fitted_composite_kernel(A, B):
        return kernel_from_spec(
            seasonal_spec(composite_theta[:-1]),
            A,
            B
        )

    # Predict future observations.
    composite_mu, composite_var = gp_predict(
        t_train,
        y_train_centered,
        t_test,
        fitted_composite_kernel,
        composite_noise_var
    )

    # Score predictions using latent variance plus observation noise.
    composite_scores = predictive_scores(
        y_test_centered,
        composite_mu,
        composite_var + composite_noise_var
    )

    # -----------------------------------------------------
    # Step 4: Assemble and return the results.
    # -----------------------------------------------------

    # Add fitted hyperparameters, LML and predictions
    # on the original target scale.
    se_scores.update({
        "lml": float(se_lml),
        "theta": se_theta.copy(),
        "mean": se_mu + y_bar
    })

    composite_scores.update({
        "lml": float(composite_lml),
        "theta": composite_theta.copy(),
        "mean": composite_mu + y_bar
    })

    return {
        "se": se_scores,
        "composite": composite_scores
    }

# Step 21 - laplace_mode
def laplace_mode(K, y, max_iter=100, tol=1e-10):
    """
    Find the posterior mode of the latent function in a
    Gaussian Process classifier using Newton's method.

    Also compute the Laplace approximation to the log
    marginal likelihood.

    Parameters
    ----------
    K : array-like, shape (N, N)
        Training Gram matrix (prior covariance matrix).
    y : array-like, shape (N,)
        Binary class labels containing only 0 and 1.
    max_iter : int, default=100
        Maximum number of Newton iterations.
    tol : float, default=1e-10
        Convergence tolerance based on the change in the
        objective function.

    Returns
    -------
    f : ndarray, shape (N,)
        Posterior mode of the latent function.

    lml : float
        Laplace-approximated log marginal likelihood.
    """

    # Convert inputs into NumPy arrays.
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Validate the covariance matrix.
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")

    N = K.shape[0]

    if N == 0:
        raise ValueError("K must not be empty.")

    if len(y) != N:
        raise ValueError("The number of labels must match K.")

    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(y)):
        raise ValueError("K and y must contain finite values.")

    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels y must contain only 0 and 1.")

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if tol < 0:
        raise ValueError("tol must be non-negative.")

    # Initialize the latent function at zero.
    f = np.zeros(N, dtype=float)

    # Store the previous objective value for convergence checking.
    previous_psi = None

    # ---------------------------------------------------------
    # Newton iterations.
    # ---------------------------------------------------------

    for iteration in range(max_iter):

        # Step 1: Compute sigmoid probabilities stably.
        pi = np.exp(-np.logaddexp(0.0, -f))

        # Compute W = pi * (1 - pi).
        W = pi * (1.0 - pi)

        # Square root of W.
        s = np.sqrt(W)

        # Step 2: Construct B = I + sqrt(W) K sqrt(W).
        B = np.eye(N) + (s[:, None] * K) * s[None, :]

        # Compute the lower Cholesky factor of B.
        L = np.linalg.cholesky(B)

        # Step 3: Compute b = W * f + (y - pi).
        b = W * f + (y - pi)

        # Compute K @ b.
        Kb = K @ b

        # Compute sqrt(W) * Kb.
        rhs = s * Kb

        # Solve L @ z = rhs.
        z = solve_triangular(
            L,
            rhs,
            lower=True
        )

        # Solve L.T @ u = z.
        u = solve_triangular(
            L.T,
            z,
            lower=False
        )

        # Compute the Newton auxiliary vector.
        a = b - s * u

        # Step 4: Update the latent function.
        f_new = K @ a

        # Compute the Bernoulli log-likelihood stably.
        log_likelihood = np.sum(
            y * (-np.logaddexp(0.0, -f_new))
            + (1.0 - y) * (-np.logaddexp(0.0, f_new))
        )

        # Compute the objective function:
        # Psi = -0.5 * a.T @ f + log likelihood.
        psi = (
            -0.5 * (a @ f_new)
            + log_likelihood
        )

        # Update the latent function.
        f = f_new

        # Step 5: Check convergence using the objective change.
        if previous_psi is not None:
            if abs(psi - previous_psi) < tol:
                break

        previous_psi = psi

    # ---------------------------------------------------------
    # Recompute the Cholesky factor at the final latent function.
    # ---------------------------------------------------------

    pi = np.exp(-np.logaddexp(0.0, -f))

    W = pi * (1.0 - pi)

    s = np.sqrt(W)

    B = np.eye(N) + (s[:, None] * K) * s[None, :]

    L = np.linalg.cholesky(B)

    # Laplace-approximated log marginal likelihood:
    # lml = Psi - sum(log(diag(L))).
    lml = psi - np.sum(np.log(np.diag(L)))

    return f, float(lml)

# Step 22 - gpc_predict
def gpc_predict(K, y, f_hat, Ks, kss):
    """
    Compute the Laplace predictive distribution of a
    Gaussian Process classifier.

    Parameters
    ----------
    K : array-like, shape (N, N)
        Training Gram matrix.
    y : array-like, shape (N,)
        Binary training labels (0 or 1).
    f_hat : array-like, shape (N,)
        Posterior mode of the latent function.
    Ks : array-like, shape (N, N_star)
        Cross-covariance matrix between training and test inputs.
    kss : array-like, shape (N_star,)
        Prior variances at the test inputs.

    Returns
    -------
    mean : ndarray, shape (N_star,)
        Predictive latent function mean.

    var : ndarray, shape (N_star,)
        Predictive latent function variance.

    prob : ndarray, shape (N_star,)
        Predictive probability of class 1.
    """

    # Convert inputs into NumPy arrays.
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    f_hat = np.asarray(f_hat, dtype=float).reshape(-1)
    Ks = np.asarray(Ks, dtype=float)
    kss = np.asarray(kss, dtype=float).reshape(-1)

    # Validate dimensions.
    N = len(y)

    if K.shape != (N, N):
        raise ValueError("K must have shape (N, N).")

    if len(f_hat) != N:
        raise ValueError("f_hat must have the same length as y.")

    if Ks.ndim != 2 or Ks.shape[0] != N:
        raise ValueError("Ks must have shape (N, N_star).")

    N_star = Ks.shape[1]

    if len(kss) != N_star:
        raise ValueError("kss must have length N_star.")

    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only binary labels 0 and 1.")

    if not all(np.all(np.isfinite(a)) for a in (K, y, f_hat, Ks, kss)):
        raise ValueError("All inputs must contain finite values.")

    if np.any(kss < 0):
        raise ValueError("Prior variances must be non-negative.")

    # --------------------------------------------------
    # Step 1: Calculate sigmoid probabilities at f_hat.
    # --------------------------------------------------

    pi_f = np.exp(-np.logaddexp(0.0, -f_hat))

    # Calculate W = pi * (1 - pi).
    W = pi_f * (1.0 - pi_f)

    # Calculate the square root of W.
    s = np.sqrt(W)

    # --------------------------------------------------
    # Step 2: Construct B and compute its Cholesky factor.
    #
    # B = I + diag(s) @ K @ diag(s)
    # --------------------------------------------------

    B = np.eye(N) + (s[:, None] * K) * s[None, :]

    L = np.linalg.cholesky(B)

    # --------------------------------------------------
    # Step 3: Calculate the predictive latent mean.
    #
    # mean = Ks.T @ (y - pi_f)
    # --------------------------------------------------

    mean = Ks.T @ (y - pi_f)

    # --------------------------------------------------
    # Step 4: Calculate the predictive latent variance.
    #
    # u = L^(-1) @ (s[:, None] * Ks)
    # var = kss - sum(u ** 2, axis=0)
    # --------------------------------------------------

    u = solve_triangular(
        L,
        s[:, None] * Ks,
        lower=True
    )

    var = kss - np.sum(u ** 2, axis=0)

    # Prevent tiny negative variances caused by rounding.
    var = np.maximum(var, 0.0)

    # --------------------------------------------------
    # Step 5: Calculate class probabilities using
    # the probit approximation to the sigmoid integral.
    #
    # kappa(v) = (1 + pi * v / 8)^(-1/2)
    # prob = sigmoid(kappa(v) * mean)
    # --------------------------------------------------

    kappa = 1.0 / np.sqrt(
        1.0 + np.pi * var / 8.0
    )

    predictive_logits = kappa * mean

    # Numerically stable sigmoid calculation.
    prob = np.exp(
        -np.logaddexp(0.0, -predictive_logits)
    )

    return mean, var, prob

# Step 23 - iris_gpc_compare
# Given Iris dataset: sepal lengths of 100 flowers.
IRIS_SEPAL_LENGTH = [
    5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,
    5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1,
    5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7,
    4.8, 5.4, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1,
    5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0,
    7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2,
    5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6,
    5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7,
    5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5,
    5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7
]

# Setosa = 0, Versicolor = 1.
IRIS_VERSICOLOR = [0] * 50 + [1] * 50


def iris_gpc_compare(
    x_query,
    length_scales=(0.25, 0.5, 1.0, 2.0, 4.0),
    taus=(0.1, 1.0, 10.0, 100.0)
):
    """
    Compare Gaussian Process classifiers using two kernel families:
    1. Squared Exponential (SE).
    2. Squared Exponential + Linear.

    Select the best hyperparameters using the Laplace
    log marginal likelihood.

    Parameters
    ----------
    x_query : array-like
        Query sepal lengths at which to predict class probabilities.

    length_scales : tuple or array-like
        Candidate SE length scales.

    taus : tuple or array-like
        Candidate linear kernel scaling parameters.

    Returns
    -------
    results : dict
        Dictionary with keys 'se' and 'se+linear'.

        Each entry contains:
        - spec: Best kernel specification.
        - lml: Best Laplace log marginal likelihood.
        - p: Predicted probabilities of class 1 at x_query.
    """

    # -----------------------------------------------------
    # Step 1: Prepare the training dataset.
    # -----------------------------------------------------

    x = as_2d(IRIS_SEPAL_LENGTH)

    y = np.asarray(IRIS_VERSICOLOR, dtype=float)

    x_query = as_2d(x_query)

    # Validate query dimensions.
    if x_query.shape[1] != x.shape[1]:
        raise ValueError(
            "Query inputs must have the same number of features as training inputs."
        )

    if x_query.shape[0] == 0:
        raise ValueError("At least one query point is required.")

    # Mean sepal length, used as the offset in the linear kernel.
    c = float(np.mean(x))

    # Ensure candidate collections can be iterated repeatedly.
    length_scales = tuple(length_scales)
    taus = tuple(taus)

    if len(length_scales) == 0:
        raise ValueError("At least one length scale is required.")

    if len(taus) == 0:
        raise ValueError("At least one tau value is required.")

    if any(not np.isfinite(l) or l <= 0 for l in length_scales):
        raise ValueError("All length scales must be positive and finite.")

    if any(not np.isfinite(tau) or tau <= 0 for tau in taus):
        raise ValueError("All tau values must be positive and finite.")

    results = {}

    # -----------------------------------------------------
    # Step 2: Evaluate the SE kernel family.
    # -----------------------------------------------------

    best_lml = -np.inf
    best_spec = None
    best_K = None
    best_f = None

    for ell in length_scales:

        # Construct the SE kernel specification.
        spec = (
            "se",
            {"length_scale": float(ell)}
        )

        # Compute the training Gram matrix.
        K = kernel_from_spec(spec, x, x)

        # Find the posterior mode and Laplace marginal likelihood.
        f_hat, lml = laplace_mode(K, y)

        # Keep the candidate with the highest LML.
        # Strict > preserves the first candidate in case of a tie.
        if lml > best_lml:
            best_lml = lml
            best_spec = spec
            best_K = K
            best_f = f_hat

    # Compute the cross-covariance matrix for the best SE kernel.
    Ks = kernel_from_spec(best_spec, x, x_query)

    # Compute prior variances at the query inputs.
    Kss = kernel_from_spec(best_spec, x_query, x_query)

    kss = np.diag(Kss)

    # Calculate predictive class probabilities.
    _, _, probabilities = gpc_predict(
        best_K,
        y,
        best_f,
        Ks,
        kss
    )

    results["se"] = {
        "spec": best_spec,
        "lml": float(best_lml),
        "p": probabilities
    }

    # -----------------------------------------------------
    # Step 3: Evaluate the SE + Linear kernel family.
    # -----------------------------------------------------

    best_lml = -np.inf
    best_spec = None
    best_K = None
    best_f = None

    # Outer loop: SE length scales.
    for ell in length_scales:

        # Inner loop: Linear kernel scaling parameters.
        for tau in taus:

            # Construct the composite kernel specification.
            spec = (
                "+",
                ("se", {"length_scale": float(ell)}),
                ("linear", {"c": c, "tau": float(tau)})
            )

            # Compute the training Gram matrix.
            K = kernel_from_spec(spec, x, x)

            # Find the posterior mode and Laplace marginal likelihood.
            f_hat, lml = laplace_mode(K, y)

            # Keep the best candidate.
            # Strict > preserves the first candidate in case of a tie.
            if lml > best_lml:
                best_lml = lml
                best_spec = spec
                best_K = K
                best_f = f_hat

    # Compute cross-covariance for the best composite kernel.
    Ks = kernel_from_spec(best_spec, x, x_query)

    # Compute prior variances at query inputs.
    Kss = kernel_from_spec(best_spec, x_query, x_query)

    kss = np.diag(Kss)

    # Calculate predictive class probabilities.
    _, _, probabilities = gpc_predict(
        best_K,
        y,
        best_f,
        Ks,
        kss
    )

    results["se+linear"] = {
        "spec": best_spec,
        "lml": float(best_lml),
        "p": probabilities
    }

    return results

# Step 24 - random_fourier_features
# ---------------------------------------------------------
# Given synthetic dataset generator for Part 6.
# ---------------------------------------------------------

def scaling_data(n_train=2000, n_test=500, noise_std=0.2, seed=0):
    """
    Generate synthetic data for evaluating scalable GP methods.

    Returns
    -------
    X : training inputs
    y : training targets
    Xt : test inputs
    yt : test targets
    """

    rng = np.random.default_rng(seed)

    X = rng.uniform(-3, 3, (n_train + n_test, 2))

    f = (
        np.sin(2 * X[:, 0])
        + 0.5 * np.cos(3 * X[:, 1])
        + 0.2 * X[:, 0] * X[:, 1]
    )

    y = f + noise_std * rng.standard_normal(len(X))

    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


# ---------------------------------------------------------
# Random Fourier Features.
# ---------------------------------------------------------

def random_fourier_features(
    D,
    M,
    length_scale,
    rng,
    kind="trig",
    orthogonal=False
):
    """
    Construct a random Fourier feature mapping for the
    Squared Exponential kernel.

    Supports:
    1. Trigonometric features.
    2. Positive exponential features.
    3. Orthogonal random features.

    Parameters
    ----------
    D : int
        Number of input dimensions.

    M : int
        Number of requested random features.

    length_scale : float
        Length scale of the Squared Exponential kernel.

    rng : numpy.random.Generator
        Random number generator.

    kind : str, default="trig"
        Feature type: "trig" or "positive".

    orthogonal : bool, default=False
        Whether to use orthogonal random features.

    Returns
    -------
    phi : callable
        A feature mapping function phi(X).
    """

    # Validate input dimensions.
    if not isinstance(D, (int, np.integer)) or D <= 0:
        raise ValueError("D must be a positive integer.")

    if not isinstance(M, (int, np.integer)) or M <= 0:
        raise ValueError("M must be a positive integer.")

    # Validate the length scale.
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be strictly positive.")

    # Validate the feature type.
    if kind not in ("trig", "positive"):
        raise ValueError("kind must be either 'trig' or 'positive'.")

    # Determine the number of random frequency vectors.
    if kind == "trig":
        n_freq = M // 2

        if n_freq == 0:
            raise ValueError("M must be at least 2 for trig features.")

    else:
        n_freq = M

    # -----------------------------------------------------
    # Step 1: Generate the frequency matrix Omega.
    # -----------------------------------------------------

    if not orthogonal:

        # Standard Gaussian random frequencies.
        # Generate all frequencies in one call.
        Omega = rng.standard_normal((n_freq, D))

    else:

        # Generate orthogonal random frequency blocks.
        blocks = []
        n_rows = 0

        while n_rows < n_freq:

            # Generate a random square Gaussian matrix.
            A = rng.standard_normal((D, D))

            # QR decomposition.
            Q, R = np.linalg.qr(A)

            # Correct the signs of Q's columns.
            signs = np.sign(np.diag(R))

            Q = Q * signs[None, :]

            # Generate independent Gaussian vector norms.
            norms = np.linalg.norm(
                rng.standard_normal((D, D)),
                axis=1
            )

            # Scale the orthogonal rows by their corresponding norms.
            block = norms[:, None] * Q

            blocks.append(block)

            n_rows += D

        # Combine blocks and retain exactly n_freq rows.
        Omega = np.vstack(blocks)[:n_freq]

    # -----------------------------------------------------
    # Step 2: Define the random feature mapping.
    # -----------------------------------------------------

    def phi(X):

        # Convert inputs into a 2D NumPy array.
        X = as_2d(X)

        # Validate the input dimensionality.
        if X.shape[1] != D:
            raise ValueError(
                "Input X must have exactly D features."
            )

        # Scale the inputs by the length scale.
        X_scaled = X / length_scale

        # Calculate the projection matrix.
        P = X_scaled @ Omega.T

        # -------------------------------------------------
        # Trigonometric random Fourier features.
        # -------------------------------------------------

        if kind == "trig":

            features = np.hstack([
                np.sin(P),
                np.cos(P)
            ])

            return features / np.sqrt(n_freq)

        # -------------------------------------------------
        # Positive random features.
        # -------------------------------------------------

        else:

            norm_sq = np.sum(
                X_scaled ** 2,
                axis=1,
                keepdims=True
            )

            features = np.exp(
                P - 0.5 * norm_sq
            )

            return features / np.sqrt(M)

    # Return the mapping with the sampled frequencies fixed.
    return phi

# Step 25 - rff_gp_predict
def rff_gp_predict(
    X,
    y,
    Xs,
    M,
    length_scale,
    signal_var,
    noise_var,
    rng,
    kind="trig",
    orthogonal=False
):
    """
    Approximate Gaussian Process regression using Random Fourier
    Features and Bayesian Linear Regression.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Training inputs.
    y : array-like, shape (N,)
        Training target values.
    Xs : array-like, shape (N_star, D)
        Test inputs.
    M : int
        Number of random features.
    length_scale : float
        Squared Exponential kernel length scale.
    signal_var : float
        Signal variance of the GP.
    noise_var : float
        Observation noise variance.
    rng : numpy.random.Generator
        Random number generator.
    kind : str, default="trig"
        Feature mapping type: "trig" or "positive".
    orthogonal : bool, default=False
        Whether to use orthogonal random features.

    Returns
    -------
    mean : ndarray, shape (N_star,)
        Predictive posterior mean.
    var : ndarray, shape (N_star,)
        Predictive latent function variance.
    """

    # Convert training and test inputs into 2D arrays.
    X = as_2d(X)
    Xs = as_2d(Xs)

    # Check input dimensions.
    if X.shape[1] != Xs.shape[1]:
        raise ValueError(
            "X and Xs must have the same number of features."
        )

    # Determine the input dimensionality.
    D = X.shape[1]

    # Step 1: Create one random Fourier feature mapping.
    # The same mapping must be used for training and test inputs.
    phi = random_fourier_features(
        D,
        M,
        length_scale,
        rng,
        kind=kind,
        orthogonal=orthogonal
    )

    # Step 2: Transform training and test inputs.
    Phi = phi(X)
    Phi_star = phi(Xs)

    # Step 3: Construct the weight prior covariance matrix.
    # Use the actual feature dimension, which can differ from M
    # when M is odd and trigonometric features are used.
    feature_dim = Phi.shape[1]

    Sigma_w = signal_var * np.eye(feature_dim)

    # Step 4: Perform Bayesian Linear Regression in feature space.
    mean, var = bayes_linreg_predict(
        Phi,
        y,
        Phi_star,
        Sigma_w,
        noise_var
    )

    return mean, var

