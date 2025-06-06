"""
This module defines a TensorFlow-based approach for fitting noisy synthetic data 
generated from a cubic polynomial model, which is used in the analysis of dark photon 
cross section ratios. It includes:

- A data generator for noisy cubic data
- A Keras-based `CubicFitModel` for learning the polynomial coefficients
- A training loop using the Adam optimizer
- Output of predictions to file
- Visualization comparing the learned fit with noisy and true data

This script is meant to serve as a replacement for `scipy.optimize.curve_fit` in noisy 
polynomial regression tasks, particularly within the context of physics simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

def generate_noisy_ratio_data(mass_points: np.ndarray, coeffs: list[float], noise_std: float = 0.05):
    """
    Generate noisy synthetic ratio data from a true cubic polynomial model.

    Parameters:
        mass_points (np.ndarray): Array of mass values in GeV
        coeffs (list[float]): True polynomial coefficients [a, b, c, d]
        noise_std (float): Standard deviation of Gaussian noise

    Returns:
        tuple: true values and noisy observations
    """
    x = mass_points * 1000  # Convert to MeV
    y_true = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3
    noise = np.random.normal(0, noise_std, size=mass_points.shape)
    return y_true, y_true + noise

# Define the TensorFlow model for cubic polynomial fitting
class CubicFitModel(Model):
    def __init__(self):
        super().__init__()
        # Initialize trainable variables for the cubic coefficients
        self.a = tf.Variable(1.0, dtype=tf.float32)
        self.b = tf.Variable(-0.01, dtype=tf.float32)
        self.c = tf.Variable(0.0002, dtype=tf.float32)
        self.d = tf.Variable(-1e-6, dtype=tf.float32)

    def call(self, x):
        # Cubic polynomial function
        return self.a + self.b * x + self.c * x**2 + self.d * x**3

if __name__ == "__main__":
    # True model parameters
    true_coeffs = [1.0, -0.01, 0.0002, -1e-6]

    # Generate mass values (GeV) and convert to MeV
    mass_points = np.linspace(0.001, 0.1, 50)
    x_data = mass_points * 1000

    # Generate true and noisy ratio values
    y_true, y_noisy = generate_noisy_ratio_data(mass_points, true_coeffs)

    # Convert data to TensorFlow tensors
    x_tf = tf.convert_to_tensor(x_data, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y_noisy, dtype=tf.float32)

    # Instantiate model and optimizer
    model = CubicFitModel()
    optimizer = Adam(learning_rate=0.01)

    # Define loss function as mean squared error
    loss_fn = lambda: tf.reduce_mean((model(x_tf) - y_tf) ** 2)

    # Perform training for 500 epochs
    for epoch in range(500):
        optimizer.minimize(loss_fn, model.trainable_variables)

    # Generate predictions from trained model
    y_pred = model(x_tf).numpy()

    # Save fitted values to file
    np.savetxt("synthetic_ratio_data_tf.txt", np.column_stack((mass_points, y_pred)),
               header="mA [GeV]\tTF_Ratio", fmt="%.6f\t%.6e")

    # Plot true function, noisy data, and TensorFlow fit
    plt.figure(figsize=(8, 5))
    plt.plot(x_data, y_true, label="True Curve", linestyle="--")
    plt.scatter(x_data, y_noisy, label="Noisy Data", alpha=0.7)
    plt.plot(x_data, y_pred, label="TF Fit", color='red')
    plt.xlabel("$m_{A'}$ [MeV]")
    plt.ylabel("Cross Section Ratio")
    plt.title("TensorFlow Fit vs True and Noisy Data")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("synthetic_ratio_plot_comparison.png")
    plt.savefig("synthetic_ratio_plot_tf.pdf")
    plt.show()
