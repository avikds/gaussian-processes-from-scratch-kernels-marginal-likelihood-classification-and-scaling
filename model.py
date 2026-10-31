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

