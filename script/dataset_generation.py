"""
Data-driven error source identification for metrology digital twin: example on Virtual CMM
This script belongs to the EPM ViDit project 22DIT01, task 2.1.6.
Author: Gengxiang CHEN, USPN, gengxiang.chen@univ-paris13.fr

The Virtual CMM simulation code and parameters configuration comes from:
author: Marcel van Dijk, VSL, mvandijk@vsl.nl
"""

import numpy as np
import pandas as pd
import json
import os

def load_settings(settings_path="default_settings_CMM_benchmark.json"):
    with open(settings_path, "r") as file:
        settings = json.load(file)
    return settings

def generate_angles(n_points):
    return np.linspace(0, 2 * np.pi, n_points, endpoint=False)

def generate_circle(radius, center, angles, lobes=0, circularity=0, lobe_angle=0):
    radii = radius + circularity / 2 * np.cos(lobes * angles + lobe_angle) * (lobes > 0)
    x = radii * np.cos(angles) + center[0]
    y = radii * np.sin(angles) + center[1]
    return np.vstack((x, y)).T

def add_errors(data, squareness, scale_x, scale_y):
    squareness_mat = np.array([[1, squareness], [0, 1]])
    scale_mat = np.array([[1 + scale_x, 0], [0, 1 + scale_y]])
    error_mat = np.dot(squareness_mat, scale_mat)
    return np.dot(data, error_mat)

def add_noise(data, stdev):
    noise = np.random.normal(0, stdev, data.shape)
    return data + noise

def generate_measurement_data(settings, squareness, scale_x, scale_y, add_noise_flag=True):

    X_info = settings["X_info"]
    Y_info = settings["Y_info"]
    # Z_info = settings["Z_info"]

    n_points = X_info["NR_DATA_POINTS"]
    noise_stdev = X_info["NOISE_STDEV"]
    radius = Y_info["RADIUS"]["Args"]["Mean"]
    center = [Y_info["CENTER_X"]["Args"]["Mean"], Y_info["CENTER_Y"]["Args"]["Mean"]]
    lobes = X_info["NR_OF_LOBES"]
    circularity = X_info["CIRCULARITY"]
    lobe_angle = X_info["LOBE_ANGLE"]

    # Generate measurement data of circles
    angles = generate_angles(n_points)
    pure_data = generate_circle(radius, center, angles, lobes, circularity, lobe_angle)

    # add some errors
    error_data = add_errors(pure_data, squareness, scale_x, scale_y)

    if add_noise_flag:
        noisy_data = add_noise(error_data, noise_stdev)
    else:
        noisy_data = error_data

    return noisy_data

def generate_dataset(settings, n_samples, squareness_params, scale_x_params, scale_y_params, label):
    data = []
    for _ in range(n_samples):
        squareness = np.random.normal(squareness_params["mean"], squareness_params["std"])
        scale_x = np.random.normal(scale_x_params["mean"], scale_x_params["std"])
        scale_y = np.random.normal(scale_y_params["mean"], scale_y_params["std"])
        measurements = generate_measurement_data(settings, squareness, scale_x, scale_y)
        data.append({
            "measurements": measurements,
            "squareness": squareness,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "label": label
        })
    return data

def save_dataset(dataset, output_path):
    os.makedirs(output_path, exist_ok=True)
    all_records = []
    for idx, sample in enumerate(dataset):
        df = pd.DataFrame(sample["measurements"], columns=["X", "Y"])
        file_name = f"sample_{idx}.csv"
        df.to_csv(os.path.join(output_path, file_name), index=False)
        all_records.append({
            "file_name": file_name,
            "squareness": sample["squareness"],
            "scale_x": sample["scale_x"],
            "scale_y": sample["scale_y"],
            "label": sample["label"]
        })
    summary_df = pd.DataFrame(all_records)
    summary_df.to_csv(os.path.join(output_path, "summary.csv"), index=False)


if __name__ == "__main__":

    settings = load_settings("default_settings_CMM.json")

    # The normal and abnormal data are generated with the different gaussian distributions
    n_samples = 200
    normal_params = {"mean": 0, "std": 5e-5}
    abnormal_params = {"mean": 2e-4, "std": 5e-5}

    # Generate measurement data with different labels
    normal_data = generate_dataset(settings, n_samples, normal_params, normal_params, normal_params, label=0)
    squareness_data = generate_dataset(settings, n_samples, abnormal_params, normal_params, normal_params, label=1)
    scale_x_data = generate_dataset(settings, n_samples, normal_params, abnormal_params, normal_params, label=2)
    scale_y_data = generate_dataset(settings, n_samples, normal_params, normal_params, abnormal_params, label=3)

    # Combine all the data
    dataset = normal_data + squareness_data + scale_x_data + scale_y_data

    # Save
    output_path = "../data"
    save_dataset(dataset, output_path)

    print(f"Data saved to {output_path}")
