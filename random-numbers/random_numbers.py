# generate_random_data.py
# This script can also be imported in the main Dark Photon script using:
# from generate_random_data import generate_noisy_ratio_data
"""
This script generates synthetic noisy data for testing curve fitting routines
in the Dark Photon cross section analysis project.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_noisy_ratio_data(mass_points: np.ndarray, coeffs: list[float], noise_std: float = 0.05):
    """
    Return noisy synthetic ratio data for use in testing polynomial fits.
    """
    x = mass_points * 1000
    y_true = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3
    noise = np.random.normal(0, noise_std, size=mass_points.shape)
    return y_true + noise

if __name__ == "__main__":
    # Define true model parameters (e.g., from a cubic ratio fit)
    true_coeffs = [1.0, -0.01, 2e-4, -1e-6]

    # Generate mass values (GeV)
    mass_points = np.linspace(0.001, 0.1, 50)  # 1 MeV to 100 MeV
    x = mass_points * 1000  # Convert to MeV for polynomial scale

    # Compute true and noisy values
    y_true = true_coeffs[0] + true_coeffs[1]*x + true_coeffs[2]*x**2 + true_coeffs[3]*x**3
    np.random.seed(42)
    y_noisy = generate_noisy_ratio_data(mass_points, true_coeffs)

    # Save to file
    np.savetxt("synthetic_ratio_data.txt", np.column_stack((mass_points, y_noisy)),
               header="mA [GeV]\tNoisyRatio", fmt="%.6f\t%.6e")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(x, y_true, label="True Curve", linestyle="--")
    plt.scatter(x, y_noisy, label="Noisy Data", alpha=0.7)
    plt.xlabel("$m_{A'}$ [MeV]")
    plt.ylabel("Cross Section Ratio")
    plt.title("Synthetic Noisy Ratio Data")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("synthetic_ratio_plot.pdf")
    plt.show()
