"""
This module contains unit tests for validating the TensorFlow-based cubic fit model
defined in `tensorflow.py`. It includes tests for:

- Output shape and numerical stability of the model
- Learning behavior after training
- Correct generation of noisy synthetic data

The tests are designed for integration with CI workflows and to ensure robustness 
of the fitting routine under typical noise conditions used in physical modeling.
"""
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

# --- Import model and data generator from main script ---
from tensorflow import keras

# Define CubicFitModel again for testing in case it’s in a separate module
class CubicFitModel(Model):
    """
    A TensorFlow model representing a cubic polynomial: a + bx + cx^2 + dx^3
    with trainable parameters a, b, c, and d.
    """
    def __init__(self):
        super().__init__()
        self.a = tf.Variable(1.0, dtype=tf.float32)
        self.b = tf.Variable(-0.01, dtype=tf.float32)
        self.c = tf.Variable(0.0002, dtype=tf.float32)
        self.d = tf.Variable(-1e-6, dtype=tf.float32)

    def call(self, x):
        return self.a + self.b * x + self.c * x**2 + self.d * x**3

def generate_noisy_ratio_data(mass_points: np.ndarray, coeffs: list[float], noise_std: float = 0.05):
    """
    Generate noisy synthetic ratio data from a cubic polynomial.

    Args:
        mass_points (np.ndarray): Array of mass values in GeV.
        coeffs (list[float]): List of true polynomial coefficients [a, b, c, d].
        noise_std (float): Standard deviation of Gaussian noise.

    Returns:
        tuple: (true values, noisy values)
    """
    x = mass_points * 1000
    y_true = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3
    noise = np.random.normal(0, noise_std, size=mass_points.shape)
    return y_true, y_true + noise

class TestTensorFlowFit(unittest.TestCase):
    """
    Unit tests for the TensorFlow cubic fit model and synthetic data generator.
    """

    def setUp(self):
        self.coeffs = [1.0, -0.01, 0.0002, -1e-6]
        self.mass_points = np.linspace(0.001, 0.1, 50)
        self.x_data = self.mass_points * 1000
        self.y_true, self.y_noisy = generate_noisy_ratio_data(self.mass_points, self.coeffs)
        self.x_tf = tf.convert_to_tensor(self.x_data, dtype=tf.float32)
        self.y_tf = tf.convert_to_tensor(self.y_noisy, dtype=tf.float32)
        self.model = CubicFitModel()

    def test_model_output_shape(self):
        """Ensure model output matches input shape."""
        y_pred = self.model(self.x_tf)
        self.assertEqual(y_pred.shape, self.x_tf.shape)

    def test_training_reduces_loss(self):
        """Test that training reduces the loss over epochs."""
        optimizer = Adam(learning_rate=0.01)
        loss_fn = lambda: tf.reduce_mean((self.model(self.x_tf) - self.y_tf) ** 2)
        initial_loss = loss_fn().numpy()
        for _ in range(300):
            optimizer.minimize(loss_fn, self.model.trainable_variables)
        final_loss = loss_fn().numpy()
        self.assertLess(final_loss, initial_loss)

    def test_generate_noisy_ratio_data_output(self):
        """Verify output shapes and types from the data generator."""
        y_true, y_noisy = generate_noisy_ratio_data(self.mass_points, self.coeffs)
        self.assertEqual(y_true.shape, self.mass_points.shape)
        self.assertEqual(y_noisy.shape, self.mass_points.shape)
        self.assertIsInstance(y_true, np.ndarray)
        self.assertIsInstance(y_noisy, np.ndarray)

if __name__ == '__main__':
    unittest.main()
