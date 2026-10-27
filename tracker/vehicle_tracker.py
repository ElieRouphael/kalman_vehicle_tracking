import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kalman.kalman_filter import KalmanFilter

df = pd.read_csv("data/example_data.csv")
measurements = df["noisy_position"].values
dt = df["time"].iloc[1] - df["time"].iloc[0]

F = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])
# Process noise. The chirp 15 sin(t^2) accelerates hard, so diag(1, 3) (the original choice) is about ten
# times too small for it: position RMSE 4.50 m against 2.93 m with the scale below (raw sensor: 5.00 m).
Q = 10 * np.array([[1, 0], [0, 3]])
R = np.array([[25]])
P = np.eye(2)
x0 = np.array([[0], [0]])

kf = KalmanFilter(F, H, Q, R, P, x0)
estimates = []
for z in measurements:
    kf.predict()
    est = kf.update(np.array([[z]]))
    estimates.append(est[0, 0])

plt.plot(df['time'], df['true_position'], label='True Position')
plt.plot(df['time'], df['noisy_position'], label='Measured Position', alpha=0.5)
plt.plot(df['time'], estimates, label='Kalman Estimate', linestyle='--')
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
print(f"position RMSE: measurements {rmse(df['noisy_position'], df['true_position']):.3f} m, Kalman {rmse(estimates, df['true_position']):.3f} m")
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()
plt.show()
