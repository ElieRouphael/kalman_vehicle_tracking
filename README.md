# Kalman Vehicle Tracking System

Estimate vehicle position from noisy sensor data using a Kalman filter: a constant-velocity model, a position
sensor with 5 m of noise, and the twenty-line filter in `kalman/kalman_filter.py`.

```bash
pip install -r requirements.txt
python data/simulate_data.py        # regenerates data/example_data.csv (seeded)
python tracker/vehicle_tracker.py   # runs the filter and plots the result
```

An interactive walk-through of this repository, with the filter running in the browser, is chapter 01 of
https://elierouphael.github.io/tutorials/kalman-jax/.

## Corrections (October 2026 review)

- `data/simulate_data.py` wrote a `true_velocity` of `2t` and a constant `acceleration` of 2 next to a position
  `15 sin(t^2)`, whose velocity is `30 t cos(t^2)`; the columns now derive from the position, and the generator is
  seeded so the CSV can be regenerated.
- The process noise `diag(1, 3)` was five to ten times too small for this motion. On the CSV as first committed,
  which chapter 01 of the course uses, the filter's position RMSE was 4.50 m against a raw-sensor 5.00 m, and 2.93 m
  at ten times the scale. On the regenerated `data/example_data.csv` the same three numbers are 4.49, 4.89 and
  2.99 m. The tracker now uses the larger value and prints both RMSEs.
  No constant-velocity `Q` makes the filter follow the chirp near the end of the run, where it oscillates at 3 Hz;
  that is a model limitation, not a tuning one.
- The filter uses a linear solve instead of `np.linalg.inv` and the Joseph-form covariance update.
