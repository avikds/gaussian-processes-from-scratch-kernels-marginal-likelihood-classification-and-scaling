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

