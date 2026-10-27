import numpy as np
import pandas as pd


def generate_vehicle_data(T=10, dt=0.01, seed=0):
    """Simulate a vehicle whose position is the chirp 15 sin(t^2), with a noisy position sensor (sd 5 m)
    and a noisy velocity sensor (sd 1 m/s). Velocity and acceleration are the derivatives of the position.
    Seeded so that data/example_data.csv can be regenerated."""
    rng = np.random.default_rng(seed)
    time = np.arange(0, T, dt)
    true_position = 15 * np.sin(time**2)
    true_velocity = 30 * time * np.cos(time**2)
    acceleration = 30 * np.cos(time**2) - 60 * time**2 * np.sin(time**2)
    noisy_position = true_position + rng.normal(0, 5, size=time.shape)
    noisy_velocity = true_velocity + rng.normal(0, 1, size=time.shape)

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
