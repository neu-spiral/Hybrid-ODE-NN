import numpy as np
import cubaturetf as cb
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
tf.random.set_seed(0)
import pdb

class CubatureFilter:

    def __init__(self, f, h, x0, P0, Q0, R0, s_dim, reg_const=0.01):
        """
        class CubatureFilter
        :param f: State transition function (nonlinear function describing state evolution)
        :param h: Measurement function (nonlinear function mapping state to measurements)
        :param x0: Initial state vector
        :param P0: Initial state covariance matrix
        :param Q0: Process noise covariance matrix
        :param R0: Measurement noise covariance matrix
        :param reg_const: Regularization constant for numerical stability
        """
        self.f = f  # State transition function
        
        # Function to compute the tensor product of the state transition function output
        def f_x_u(x, u):
            temp = self.f(x, u)
            return tf.tensordot(temp, temp, axes=0)
        self.fft = f_x_u
        self.h = h  # Measurement function
        
        # Function to compute the tensor product of the measurement function output
        def h_x(x):
            temp = self.h(x)
            return tf.tensordot(temp, temp, axes=0)
        self.hht = h_x
        self.x = x0  # Initial state
        self.P = P0  # Initial state covariance
        self.Q = Q0  # Process noise covariance
        self.R = R0  # Measurement noise covariance
        self.s_dim = s_dim  # State dimension
        self.dim_x = len(self.x)  # Length of state vector
        self.cov_type = "diag"  # Covariance type (default: diagonal)
        # Determine measurement dimension based on R0 (if callable, evaluate it to get shape)
        if callable(R0):
            self.dim_y = R0().shape[0]
        else:
            self.dim_y = R0.shape[0]
        self.n_cubature_points = 2*self.dim_x  # Number of cubature points (2 * state dimension)
        self.reg_mat = reg_const*tf.keras.backend.eye(len(x0)) # Regularization matrix for stability

    def predict(self, sigmoid, u=None):
        # Prediction step of the cubature filter
        x_pred, cubature_points , P_pred_temp = cb.cubature_int_f(self.f, self.x, self.P, return_cubature_points=True, u=u)
        P_pred = P_pred_temp - tf.tensordot(x_pred, x_pred,axes=0) + self.Q  # Update covariance with process
        return x_pred, P_pred

    def update(self, y, sigmoid, u=None):
        # Update step of the cubature filter
        x_pred, P_pred = self.predict(sigmoid=sigmoid,u=u)
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.dim_x, self.n_cubature_points)
        # Generate predicted measurement cubature points
        y_cubature_points = tf.constant([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
        y_pred = tf.math.reduce_mean(y_cubature_points, axis=0)# Predicted measurement mean
        
        # Determine measurement noise covariance
        if callable(self.R):
            R = self.R(x_pred)
        else:
            R = self.R
        # Compute innovation covariance
        P_yy = cb.cubature_int(self.hht, x_pred, P_pred, cubature_points=x_cubature_points) - tf.tensordot(y_pred, y_pred,axes = 0) + R
        P_xy = tf.math.reduce_mean([tf.tensordot(x_cubature_points[i], y_cubature_points[i],axes=0) for i in range(self.n_cubature_points)], axis=0) - tf.tensordot(x_pred, y_pred,axes=0)
        # Calculate Kalman gain
        Kg = tf.transpose(tf.linalg.matvec(P_xy, cb.inv_pd_mat(P_yy)))
        
        # Update state and covariance
        self.x = x_pred + tf.linalg.matvec(Kg, (y - y_pred))# Update state with measurement residual
        self.P = P_pred - tf.linalg.matmul(tf.linalg.matmul(Kg, P_yy), tf.transpose(Kg)) # Update covariance
        self.P = tf.Variable(self.P)# Store updated covariance as a variable
        return y_pred, (y - y_pred)

    def test_forward(self, y, u=None):
        # Test forward step for evaluating predictions
        x_pred, cubature_points , P_pred_temp = cb.cubature_int_f(self.f, self.x[-self.s_dim:], self.P[-self.s_dim:,-self.s_dim:], return_cubature_points=True, u=u[0:self.s_dim*2,:])
        P_pred = P_pred_temp - tf.tensordot(x_pred, x_pred,axes=0) + self.Q
        
        # Generate cubature points and predicted measurement
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.s_dim, 2*self.s_dim)#instead of self.n_cubature_points
        y_cubature_points = tf.constant([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
        y_pred = tf.math.reduce_mean(y_cubature_points, axis=0)
        
        # Determine measurement noise covariance
        if callable(self.R):
            R = self.R(x_pred)
        else:
            R = self.R
        P_yy = cb.cubature_int(self.hht, x_pred, P_pred, cubature_points=x_cubature_points) - tf.tensordot(y_pred, y_pred,axes = 0) + R
        P_xy = tf.math.reduce_mean([tf.tensordot(x_cubature_points[i], y_cubature_points[i],axes=0) for i in range(self.n_cubature_points)], axis=0) - tf.tensordot(x_pred, y_pred,axes=0)
        Kg = tf.transpose(tf.linalg.matvec(P_xy, cb.inv_pd_mat(P_yy)))
        self.x = x_pred + tf.linalg.matvec(Kg, (y - y_pred))
        self.P = P_pred - tf.linalg.matmul(tf.linalg.matmul(Kg, P_yy), tf.transpose(Kg))
        self.P = tf.Variable(self.P)
        return y_pred, (y - y_pred)

    def open_loop_forward(self, y, u=None):
        # Open-loop forward step (without correction)
        x_pred, cubature_points = cb.cubature_int(self.f, self.x[-self.s_dim:], self.P[-self.s_dim:,-self.s_dim:], return_cubature_points=True, u=u)
        P_pred = cb.cubature_int(self.fft, self.x[-self.s_dim:], self.P[-self.s_dim:,-self.s_dim:], cubature_points=cubature_points, u=u) - np.outer(x_pred, x_pred) + self.Q[-self.s_dim:,-self.s_dim:]
        x_cubature_points = cb.gen_cubature_points(x_pred, P_pred, self.s_dim, 2*self.s_dim)#instead of self.n_cubature_points

        # Generate predicted measurement cubature points
        y_cubature_points = np.array([self.h(x_cubature_points[i]) for i in range(len(x_cubature_points))])
        y_pred = np.mean(y_cubature_points, axis=0)

        # Update state and covariance without correction term
        self.x[-self.s_dim:] = x_pred
        self.P[-self.s_dim:,-self.s_dim:] = P_pred
        #no correction term
        return y_pred, (y - y_pred)

    def reset_states(self):
        # Reset the last s_dim elements of the state vector to zero
        self.x[-self.s_dim:] *= 0
        
    def get_states(self):
        # Get the current state vector
        return self.x
