import numpy as np


class KalmanFilter:
    """Linear Kalman filter. The gain is computed with a linear solve rather than an explicit inverse, and the
    covariance update uses the Joseph form, which stays symmetric and positive semi-definite in floating point."""

    def __init__(self, F, H, Q, R, P, x0):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x                                  # innovation
        S = self.H @ self.P @ self.H.T + self.R                  # innovation covariance
        K = np.linalg.solve(S, self.H @ self.P.T).T              # gain, P H^T S^-1 without inv()
        self.x = self.x + K @ y
        I_KH = np.eye(self.P.shape[0]) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T       # Joseph form
        return self.x
