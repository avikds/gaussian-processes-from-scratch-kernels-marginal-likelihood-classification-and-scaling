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

