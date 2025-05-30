# This script replaces the polynomial fit using SciPy with a TensorFlow model.
# It demonstrates how TensorFlow can be used for fitting noisy synthetic data.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define a custom TensorFlow model for cubic polynomial
class CubicFitModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.a = tf.Variable(1.0, dtype=tf.float32)
        self.b = tf.Variable(-0.01, dtype=tf.float32)
        self.c = tf.Variable(0.0002, dtype=tf.float32)
        self.d = tf.Variable(-1e-6, dtype=tf.float32)

    def call(self, x):
        return self.a + self.b * x + self.c * tf.pow(x, 2) + self.d * tf.pow(x, 3)

# Generate synthetic noisy data
def generate_noisy_ratio_data(mass_points: np.ndarray, coeffs: list[float], noise_std: float = 0.05):
    x = mass_points * 1000
    y_true = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3
    noise = np.random.normal(0, noise_std, size=mass_points.shape)
    return x, y_true + noise, y_true

# Train model using TensorFlow gradient descent
def fit_tensorflow_model(x_data, y_data, epochs=500):
    x_tf = tf.convert_to_tensor(x_data, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y_data, dtype=tf.float32)
    model = CubicFitModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x_tf) - y_tf))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return model

if __name__ == "__main__":
    true_coeffs = [1.0, -0.01, 2e-4, -1e-6]
    mass_points = np.linspace(0.001, 0.1, 50)
    x_vals, y_noisy, y_true = generate_noisy_ratio_data(mass_points, true_coeffs)

    np.random.seed(42)
    model = fit_tensorflow_model(x_vals, y_noisy)
    y_fit = model(tf.convert_to_tensor(x_vals, dtype=tf.float32)).numpy()

    np.savetxt("synthetic_ratio_data_tf.txt", np.column_stack((mass_points, y_fit)),
               header="mA [GeV]\tFittedRatio", fmt="%.6f\t%.6e")

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_true, label="True Curve", linestyle="--")
    plt.scatter(x_vals, y_noisy, label="Noisy Data", alpha=0.7)
    plt.plot(x_vals, y_fit, label="TensorFlow Fit", color='red')
    plt.xlabel("$m_{A'}$ [MeV]")
    plt.ylabel("Cross Section Ratio")
    plt.title("TensorFlow Fit vs Noisy Data")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("synthetic_ratio_plot_tf.pdf")
    plt.show()
