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
Q = np.array([[1, 0], [0, 3]])
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
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()
plt.show()
