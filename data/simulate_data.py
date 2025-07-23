import numpy as np
import pandas as pd

def generate_vehicle_data(T=10, dt=0.01):
    time = np.arange(0, T, dt)
    true_position = 15*np.sin(time**2)
    true_velocity = 2.0 * time
    acceleration = np.full_like(time, 2.0)
    noisy_position = true_position + np.random.normal(0, 5, size=time.shape)
    noisy_velocity = true_velocity + np.random.normal(0, 1, size=time.shape)

    df = pd.DataFrame({
        'time': time,
        'acceleration': acceleration,
        'true_position': true_position,
        'noisy_position': noisy_position,
        'true_velocity': true_velocity,
        'noisy_velocity': noisy_velocity
    })
    df.to_csv('data/example_data.csv', index=False)
    print("Data saved to data/example_data.csv")

if __name__ == "__main__":
    generate_vehicle_data()
