import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import os

def ar1(length, phi, noise_std):
    x = np.zeros(length)
    for t in range(1, length):
        x[t] = phi * x[t - 1] + np.random.normal(0, noise_std)
    return x

# Original function: single drift generation
def generate_drift_dataset(
    length=4000,
    drift_point=2000,
    gradual=False,
    gradual_duration=1000,
    seed=42,
    output_file="drift_dataset.csv"
):
    np.random.seed(seed)

    start_date = datetime(2000, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(length)]

    # Two generators (could add more for variety)
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

    df = pd.DataFrame({
        "Date": dates,
        "Value": values
    })

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data/", output_file)
    df.to_csv(output_path, index=False)

    return output_path

def generate_multi_drift_dataset(
    total_length=7000,
    drift_configs=None,
    seed=42,
    output_file="multi_drift_dataset.csv"
):
    np.random.seed(seed)

    if drift_configs is None:
        drift_configs = [
            {"type": "sudden", "point": 2000},
            {"type": "gradual", "point": 4000, "duration": 800},
        ]

    start_date = datetime(2000, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(total_length)]

    gen_params = [
        {"phi": 0.8, "noise_std": 0.3},
        {"phi": -0.5, "noise_std": 0.7},
        {"phi": 0.3, "noise_std": 0.2},
        {"phi": 0.9, "noise_std": 0.5},
        {"phi": -0.2, "noise_std": 0.4},
    ]

    values = np.zeros(total_length)
    current_gen_index = 0
    drift_configs = sorted(drift_configs, key=lambda x: x["point"])

    current_start = 0

    for idx, drift in enumerate(drift_configs):
        drift_point = drift["point"]
        drift_type = drift["type"]

        segment_end = drift_point if drift_type == "sudden" else min(drift_point + drift.get("duration", 500), total_length)

        # Generate two generators for transition
        length = segment_end - current_start
        gen1 = ar1(length, **gen_params[current_gen_index % len(gen_params)])
        current_gen_index += 1
        gen2 = ar1(length, **gen_params[current_gen_index % len(gen_params)])

        if drift_type == "sudden":
            values[current_start:drift_point] = gen1[:drift_point - current_start]
            values[drift_point:segment_end] = gen2[:segment_end - drift_point]
        elif drift_type == "gradual":
            for t in range(current_start, segment_end):
                if t < drift_point:
                    values[t] = gen1[t - current_start]
                elif drift_point <= t < segment_end:
                    alpha = (t - drift_point) / (segment_end - drift_point)
                    values[t] = (1 - alpha) * gen1[t - current_start] + alpha * gen2[t - current_start]
        else:
            raise ValueError(f"Unknown drift type: {drift_type}")

        current_start = segment_end

    # Fill the rest of the series after the last drift
    if current_start < total_length:
        tail_length = total_length - current_start
        final_gen = ar1(tail_length, **gen_params[current_gen_index % len(gen_params)])
        values[current_start:] = final_gen

    df = pd.DataFrame({
        "Date": dates,
        "Value": values
    })

    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", output_file)
    df.to_csv(output_path, index=False)

    return output_path

# # === Generate all datasets ===

# # Original: Sudden drift at 5500
# generate_drift_dataset(
#     length=7000,
#     drift_point=5500,
#     gradual=False,
#     seed=42,
#     output_file="SUD.csv"
# )

# # Original: Gradual drift at 5500 (duration 600)
# generate_drift_dataset(
#     length=7000,
#     drift_point=5500,
#     gradual=True,
#     gradual_duration=600,
#     seed=42,
#     output_file="GRD.csv"
# )

# New: Gradual drift at 5500 (duration 600) + Sudden drift at 6600
generate_multi_drift_dataset(
    total_length=7000,
    drift_configs=[
        {"type": "gradual", "point": 5500, "duration": 600},
        {"type": "sudden", "point": 6600}
    ],
    seed=42,
    output_file="GRD_SUD.csv"
)