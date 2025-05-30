"""
This script fits noisy cubic ratio data using TensorFlow instead of curve_fit,
and compares it visually with the true function and noisy data.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

def generate_noisy_ratio_data(mass_points: np.ndarray, coeffs: list[float], noise_std: float = 0.05):
    x = mass_points * 1000
    y_true = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3
    noise = np.random.normal(0, noise_std, size=mass_points.shape)
    return y_true, y_true + noise

# Define the model
class CubicFitModel(Model):
    def __init__(self):
        super().__init__()
        self.a = tf.Variable(1.0, dtype=tf.float32)
        self.b = tf.Variable(-0.01, dtype=tf.float32)
        self.c = tf.Variable(0.0002, dtype=tf.float32)
        self.d = tf.Variable(-1e-6, dtype=tf.float32)

    def call(self, x):
        return self.a + self.b * x + self.c * x**2 + self.d * x**3

if __name__ == "__main__":
    true_coeffs = [1.0, -0.01, 0.0002, -1e-6]
    mass_points = np.linspace(0.001, 0.1, 50)
    x_data = mass_points * 1000
    y_true, y_noisy = generate_noisy_ratio_data(mass_points, true_coeffs)

    x_tf = tf.convert_to_tensor(x_data, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y_noisy, dtype=tf.float32)

    model = CubicFitModel()
    optimizer = Adam(learning_rate=0.01)
    loss_fn = lambda: tf.reduce_mean((model(x_tf) - y_tf) ** 2)

    for epoch in range(500):
        optimizer.minimize(loss_fn, model.trainable_variables)

    y_pred = model(x_tf).numpy()
    np.savetxt("synthetic_ratio_data_tf.txt", np.column_stack((mass_points, y_pred)),
               header="mA [GeV]\tTF_Ratio", fmt="%.6f\t%.6e")

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
