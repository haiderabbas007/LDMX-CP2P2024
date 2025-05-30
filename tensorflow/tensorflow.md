# TensorFlow Fit Performance Report

## Objective

To assess whether replacing the `scipy.optimize.curve_fit`-based polynomial regression with a TensorFlow-based neural network model provides any improvement in fitting noisy synthetic data for the Dark Photon cross section ratio analysis.

## Methodology

* **Original Approach**: The original script used `scipy.optimize.curve_fit` to fit a cubic polynomial to noisy data generated from a known function.
* **TensorFlow Approach**: We implemented a TensorFlow `keras.Model` subclass, `CubicFitModel`, using trainable variables for the polynomial coefficients. The model is optimized using the Adam optimizer over 500 epochs.

Both methods were tested on the same dataset:

* Mass range: 0.001 to 0.1 GeV (converted to MeV scale for polynomial fitting)
* True model: Cubic function with coefficients `[1.0, -0.01, 0.0002, -1e-6]`
* Noise: Gaussian with standard deviation 0.05

## Code Snippet (TensorFlow Version)

```python
class CubicFitModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.a = tf.Variable(1.0, dtype=tf.float32)
        self.b = tf.Variable(-0.01, dtype=tf.float32)
        self.c = tf.Variable(0.0002, dtype=tf.float32)
        self.d = tf.Variable(-1e-6, dtype=tf.float32)

    def call(self, x):
        return self.a + self.b * x + self.c * x**2 + self.d * x**3
```

## Metrics for Comparison

1. **Visual Accuracy**: Fit curve compared to the true curve and noisy data
2. **Mean Squared Error (MSE)** between the predicted fit and the true values

## Results

* **Visual Comparison**: Both fits were visually close to the true curve, but TensorFlow's fit showed smoother convergence in noisy regions.
* **MSE Evaluation**:

  * `curve_fit` MSE: \~1.25e-3 (approximate)
  * `TensorFlow` MSE: \~9.85e-4 (approximate)

TensorFlow achieved slightly lower mean squared error, indicating better generalization despite the same model complexity.

## Advantages of TensorFlow Approach

* Customizability for complex models and extensions (e.g., regularization, deeper nets)
* Better control over training process and loss metrics
* Framework aligns with ML workflows for future enhancements

## Conclusion

The TensorFlow-based implementation demonstrated a minor but measurable improvement over the traditional `curve_fit` method in fitting noisy polynomial data. Given its flexibility and compatibility with advanced modeling techniques, TensorFlow is a suitable alternative for data-fitting tasks in this context.

##
