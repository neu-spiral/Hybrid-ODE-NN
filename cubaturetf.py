import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import time
import pdb

debug_mod = False
def cubature_int_f(func, mean, cov, return_cubature_points=False, cubature_points=None, u=None, sqr=False):
    """
    Computes the cubature integral of the type "\int func(x) p(x) dx", where func is an arbitrary function of x, p(x) is
    considered a Gaussian distribution with mean given by 'mean', and covariance matrix 'cov'. The integral is performed
    numerically as func_int = sum(func(sig_points(i)))/N
    The outputs are the mean vector ('func_int') and the cubature points matrix ('cubature_points').
        :param func: function to be integrated.
        :param mean: (d,) mean numpy array
        :param cov: (d,d) numpy array (covariance matrix)
        :param cubature_points: (d, 2*d) numpy array containing the 2*d cubature_points (this depends on the distribution p(x) if
                            it changes new points need to be generated.
        :param u: input signal in case of f(x,u) (default u=None, f(x) ).
        :param sqr: Boolean defining if running a square-root kf. If True than cov should be a lower triangular
        decomposition of the covariance matrix! That is cov = cholesky(P).
        :return: func_int
        -------------------------------------------------------------------
    e.g.
        cov = np.array([[1.2769, 0.0843], [0.0843, 1.0725]])
            mean = np.array([0.1, 2])
            fh = lambda x: x**2 + 10
            mean, cubature_points = cubature_int(fh, mean, cov)
            print(mean, cubature_points)
        -------------------------------------------------------------------
        Author: Tales Imbiriba
        Last modified by Ahmet Demirkaya in Aug 2022 for parallelizing the computations and speeding up the code.
    """
    
    d = len(mean)  # Dimension of the mean vector
    n_points = 2 * d  # Number of cubature points (2*d for Gaussian distributions)

    # Generate cubature points if they are not provided
    if cubature_points is None:
        # create cubature points;
        cubature_points = gen_cubature_points(mean, cov, d, n_points, sqr)

    # Compute the function values at the cubature points    
    int_ = func(cubature_points, u)

    # Calculate the predicted covariance matrix (P_pred_temp) by taking tensor product of function values
    P_pred_temp_arr = [tf.tensordot(x, x, axes=0) for x in int_]
    x_pred = tf.math.reduce_mean(int_, axis=0) # Mean prediction by averaging the function values
    P_pred_temp = tf.math.reduce_mean(P_pred_temp_arr, axis = 0) # Covariance prediction by averaging tensor products
    
    return x_pred, cubature_points, P_pred_temp



def cubature_int(func, mean, cov, return_cubature_points=False, cubature_points=None, u=None, sqr=False):
    """
    Computes the cubature integral of the type "\int func(x) p(x) dx", where func is an arbitrary function of x, p(x) is
    considered a Gaussian distribution with mean given by 'mean', and covariance matrix 'cov'. The integral is performed
    numerically as func_int = sum(func(sig_points(i)))/N
    The outputs are the mean vector ('func_int') and the cubature points matrix ('cubature_points').
    :param func: function to be integrated.
    :param mean: (d,) mean numpy array
    :param cov: (d,d) numpy array (covariance matrix)
    :param cubature_points: (d, 2*d) numpy array containing the 2*d cubature_points (this depends on the distribution p(x) if
                        it changes new points need to be generated.
    :param u: input signal in case of f(x,u) (default u=None, f(x) ).
    :param sqr: Boolean defining if running a square-root kf. If True than cov should be a lower triangular
    decomposition of the covariance matrix! That is cov = cholesky(P).
    :return: func_int
    -------------------------------------------------------------------
    e.g.
        cov = np.array([[1.2769, 0.0843], [0.0843, 1.0725]])
        mean = np.array([0.1, 2])
        fh = lambda x: x**2 + 10
        mean, cubature_points = cubature_int(fh, mean, cov)
        print(mean, cubature_points)
    -------------------------------------------------------------------
    Author: Tales Imbiriba
    Last modified in Jan 2021.
    """
    
    d = len(mean)  # Dimension of the mean vector
    n_points = 2 * d  # Number of cubature points (2*d for Gaussian distributions)

    # Generate cubature points if not provided
    if cubature_points is None:
        # create cubature points;
        cubature_points = gen_cubature_points(mean, cov, d, n_points, sqr)

    

    # Calculate the mean of the function over the cubature points
    int_mean = int_func(func, cubature_points, u)

    # Return the mean and optionally the cubature points
    if return_cubature_points:
        return int_mean, cubature_points
    else:
        return int_mean

def gen_cubature_points(mean, cov, d, n_points, sqr=False):
    # Initialize cubature points matrix with zeros
    cubature_points = tf.keras.backend.zeros((n_points, d))
    
    # Compute the lower triangular matrix L such that cov = L * L'
    if sqr:
        L = cov
    else:
        reg = 0.000001
        L = tf.linalg.cholesky(cov + reg * tf.keras.backend.eye(cov.shape[0]))
        # L = tf.linalg.cholesky(cov + reg * tf.keras.backend.eye(cov.shape[0]))

    num = tf.sqrt(n_points / 2)
    num = tf.cast(num, tf.float64)
    num_eye = num * tf.keras.backend.eye(d)
    xi = tf.keras.backend.concatenate((num_eye, -num_eye ), axis=1)
    xi = tf.keras.backend.transpose(xi)
    
    cubature_points = mean + tf.linalg.matvec(L, xi)
    return cubature_points

def int_func(func, cubature_points, u=None):
    # Evaluate the function at each cubature point and take the average
    if u is None:
        return tf.math.reduce_mean([func(x)    for x in cubature_points], axis=0)
    else:
        return tf.math.reduce_mean([func(x, u) for x in cubature_points], axis=0)

def inv_pd_mat(K, reg=1e-5):
    """
    Usage: inv_pd_mat(self, K, reg=1e-5)
    Invert (Squared) Positive Definite matrix using Cholesky decomposition.
    :param K: Positive definite matrix. (ndarray).
    :param reg: a regularization parameter (default: reg = 1e-6).
    :return: the inverse of K.
    """
    
    # Add small regularization term to ensure numerical stability
    # # compute inverse K_inv of K based on its Cholesky
    # decomposition L and its inverse L_inv
    K = tf.identity(K) + reg * tf.keras.backend.eye(len(K))
    # Compute the Cholesky decomposition of the regularized matrix K
    L = tf.linalg.cholesky(K)
    # Solve for the inverse of L using triangular solve
    L_inv = tf.linalg.triangular_solve(tf.transpose(L), tf.keras.backend.eye(L.shape[0]), lower=False)
    # Compute the inverse of K by taking the tensor product of L_inv with its transpose
    inv = tf.tensordot(L_inv, tf.transpose(L_inv), axes = 1)
    return inv 
    
