

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.fftpack import fft
import pandas as pd
plt.rc('font', family='arial', size=12)

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams['mathtext.fontset'] = 'stix'

def fit_circle(x, y):
    def cost(params):
        x0, y0, r = params
        return np.sum((np.sqrt((x - x0)**2 + (y - y0)**2) - r)**2)
    x_mean, y_mean = np.mean(x), np.mean(y)
    r_mean = np.mean(np.sqrt((x - x_mean)**2 + (y - y_mean)**2))
    return x_mean, y_mean, r_mean

if __name__ == "__main__":

    # Normal sample: From 0-199
    # squareness abnormal sample: From 200-399
    # scale_x abnormal sample: From 400-599
    # scale_y abnormal sample: From 600-799

    data_path = "../data/sample_0.csv"  # Replace with your data file path
    fig_name = 'sample_analysis'

    # Load data
    data = pd.read_csv(data_path)
    x = data["X"].values
    y = data["Y"].values

    # Fit a circle to the data
    x0_fit, y0_fit, r_fit = fit_circle(x, y)

    r_fit = 50
    residuals = np.sqrt((x - x0_fit) ** 2 + (y - y0_fit) ** 2) - r_fit

    # Remove mean from residuals
    residuals_zero_mean = residuals - np.mean(residuals)

    # Perform FFT on zero-mean residuals
    fft_residuals = fft(residuals_zero_mean)
    fft_magnitude = np.abs(fft_residuals)[:len(fft_residuals) // 2]
    fft_frequencies = np.fft.fftfreq(len(residuals_zero_mean))[:len(residuals_zero_mean) // 2]

    y_measured = x
    x_measured = y
    # Compute angles for measured points
    angles = np.arctan2(y_measured - y0_fit, x_measured - x0_fit)

    # Compute fitted coordinates
    x_fitted = x0_fit + r_fit * np.cos(angles)
    y_fitted = y0_fit + r_fit * np.sin(angles)

    # Compute residuals
    residuals_x = x_measured - x_fitted
    residuals_y = y_measured - y_fitted

    # Compute normal vector components
    norm_x = x_fitted - x0_fit
    norm_y = y_fitted - y0_fit
    norm_magnitude = np.sqrt(norm_x ** 2 + norm_y ** 2)
    norm_x /= norm_magnitude
    norm_y /= norm_magnitude

    # Compute normal residuals
    residuals_n = residuals_x * norm_x + residuals_y * norm_y

    # Zoom along normal direction
    zoom_factor = 2000  # Zoom factor for residuals
    x_zoomed = x_fitted + norm_x * residuals_n * zoom_factor
    y_zoomed = y_fitted + norm_y * residuals_n * zoom_factor

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # fig.suptitle("Residuals analysis", fontsize=14)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92, wspace=0.3, hspace=0.3)
    plt.rcParams["font.family"] = "Arial"

    #  Zoomed-in view
    axes[0, 0].plot(x_zoomed, y_zoomed, 'r.', markersize=2, alpha=0.7, label="Measured Data")
    axes[0, 0].scatter([x0_fit], [y0_fit], c='#f24c1d')
    circle_theta = np.linspace(0, 2 * np.pi, 1000)
    axes[0, 0].plot(r_fit * np.cos(circle_theta) + x0_fit,
                    r_fit * np.sin(circle_theta) + y0_fit,
                    color='blue', linestyle='-', label="Fitted Circle")
    axes[0, 0].set_title("(a) Zoomed-in View of Data and Fitted Center", fontsize=12)
    axes[0, 0].set_xlabel("X (mm)")
    axes[0, 0].set_ylabel("Y (mm)")
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].axis('equal')

    # Residuals from fitted circle
    angles = np.arctan2(y - y0_fit, x - x0_fit)
    axes[0, 1].scatter(angles, residuals, s=1, alpha=0.7, c='purple')
    axes[0, 1].set_title("(b) Residuals from Fitted Circle", fontsize=12)
    axes[0, 1].set_xlabel("Angle (radians)")
    axes[0, 1].set_ylabel("Residual (mm)")

    #  Histogram of residuals
    axes[0, 2].hist(residuals, bins=30, alpha=0.6, color='red', edgecolor='red')
    axes[0, 2].set_title("(c) Histogram of Residuals", fontsize=12)
    axes[0, 2].set_xlabel("Residual (mm)")
    axes[0, 2].set_ylabel("Frequency")

    #  X residuals vs. Fitted X
    axes[1, 0].scatter(x_fitted, residuals_x, s=1, alpha=0.7, c='b')
    axes[1, 0].set_title("(d) X Residuals vs. Fitted X", loc='center', fontsize=12)
    axes[1, 0].set_xlabel("Fitted X (mm)")
    axes[1, 0].set_ylabel("X Residual (mm)")

    # Y residuals vs. Fitted Y
    axes[1, 1].scatter(y_fitted, residuals_y, s=1, alpha=0.7, c='g')
    axes[1, 1].set_title("(e) Y Residuals vs. Fitted Y", loc='center', fontsize=12)
    axes[1, 1].set_xlabel("Fitted Y (mm)")
    axes[1, 1].set_ylabel("Y Residual (mm)")

    #  X residuals vs. Fitted Y (Squareness Analysis)
    axes[1, 2].scatter(y_fitted, residuals_x, s=1, alpha=0.7, c='r')
    axes[1, 2].set_title("(f) X Residuals vs. Fitted Y (Squareness)", loc='center', fontsize=12)
    axes[1, 2].set_xlabel("Fitted Y (mm)")
    axes[1, 2].set_ylabel("X Residual (mm)")

    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust layout to leave space for title
    plt.savefig('fig/' + fig_name + '.jpg')
    plt.show()




