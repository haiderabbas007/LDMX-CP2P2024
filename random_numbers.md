# Random Data Generator for Dark Photon Ratio Fits

This module provides a Python utility script for generating **synthetic noisy data** to test and validate polynomial fitting routines used in the dark photon cross section analysis framework.

---

## Purpose

The script `generate_random_data.py` produces realistic, noisy ratio data mimicking the form of cross section ratios (e.g., simulation/code, double ratio corrections). This allows you to:

* Validate curve fitting algorithms,
* Benchmark polynomial fitting procedures,
* Generate reproducible figures and debug fitting pipelines,
* Integrate synthetic test cases into your analysis.

---

## How It Works

The key function:

```python
from generate_random_data import generate_noisy_ratio_data
```

accepts an array of mass points (in GeV), a list of true polynomial coefficients, and a standard deviation for noise. It returns noisy cubic-polynomial data in MeV:

```python
def generate_noisy_ratio_data(mass_points: np.ndarray, coeffs: list[float], noise_std: float = 0.05):
    """
    Return noisy synthetic ratio data for use in testing polynomial fits.
    """
    x = mass_points * 1000  # Convert to MeV
    y_true = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3
    noise = np.random.normal(0, noise_std, size=mass_points.shape)
    return y_true + noise
```

---

## Standalone Execution

When run directly:

* Defines a truth model: `true_coeffs = [1.0, -0.01, 2e-4, -1e-6]`
* Generates 50 mass points from 1 MeV to 100 MeV
* Produces true and noisy data values
* Saves data to `synthetic_ratio_data.txt`
* Plots and saves the curve and noisy data to `synthetic_ratio_plot.pdf`

---

## Output Files

* **synthetic\_ratio\_data.txt**: Two-column tab-separated file with mass and noisy ratio
* **synthetic\_ratio\_plot.pdf**: Plot comparing the true model with noisy points

---

## Use in Larger Projects

This utility is designed to be used with dark photon cross section analyses involving:

* Polynomial ratio fitting
* Correction factor parameterization
* Yield and double ratio analysis

It supports isolated testing of fit behavior prior to applying routines to real or simulated physics data.

---
