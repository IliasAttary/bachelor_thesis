import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import os

def generate_drift_dataset(
    length=4000,
    drift_point=2000,
    gradual=False,
    gradual_duration=1000,
    seed=42,
    output_file="drift_dataset.csv"
):
    np.random.seed(seed)

    # Time index generation (assuming daily frequency)
    start_date = datetime(2000, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(length)]

    # Generators: AR(1) processes with different coefficients
    def ar1(length, phi, noise_std):
        x = np.zeros(length)
        for t in range(1, length):
            x[t] = phi * x[t-1] + np.random.normal(0, noise_std)
        return x

    # Generate full base sequences
    gen1 = ar1(length, phi=0.8, noise_std=0.3)
    gen2 = ar1(length, phi=-0.5, noise_std=0.7)

    values = np.zeros(length)

    if gradual:
        end_gradual = drift_point + gradual_duration
        for t in range(length):
            if t < drift_point:
                values[t] = gen1[t]
            elif drift_point <= t < end_gradual:
                alpha = (t - drift_point) / gradual_duration
                values[t] = (1 - alpha) * gen1[t] + alpha * gen2[t]
            else:
                values[t] = gen2[t]
    else:
        values[:drift_point] = gen1[:drift_point]
        values[drift_point:] = gen2[drift_point:]

    # Create dataframe
    df = pd.DataFrame({
        "Date": dates,
        "Value": values
    })

    # Save to CSV
    output_path = os.path.join("data/", output_file)
    df.to_csv(output_path, index=False)

    return output_path


# generate a Sudden  drift dataset
generate_drift_dataset(
    length=7000,
    drift_point=5500,
    gradual=False,
    seed=42,
    output_file="SUD.csv"
)

# generate a gradual drift dataset
generate_drift_dataset(
    length=7000,
    drift_point=5500,
    gradual=True,
    gradual_duration=600,
    seed=42,
    output_file="GRD.csv"
)