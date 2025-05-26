# Data Type Representations

This document summarizes the data types used in the implementation of the Dark Photon cross-section analysis project. The goal is to document their usage with accurate mappings to Python annotations.

---

## 1. Scalar Types

### Examples

* `mA: float` – used as the mass of the dark photon in `compute_cross_section(mA, E0, Z, A)`
* `E0: float` – beam energy passed to the same function
* `Z: int`, `A: float` – atomic number and mass number
* `options.Display: bool` – parsed from command-line or notebook config

|  Variable               | Python Type Annotation | Notes                                                                     |
| -------------------------------- | ---------------------- | ------------------------------------------------------------------------- |
| Beam energy, mass, cross section | `float`                | Used for physics inputs and model calculations (e.g. `mA`, `E0`, `sigma`) |
| Integer flags, loop counters     | `int`                  | Used for index-based loops and configuration (e.g. `Z`, `energy level`)   |
| Boolean flags (`Display`, etc.)  | `bool`                 | Used in option flags from CLI / notebook configuration                    |

---

## 2. Arrays and Collections

### Examples

* `mass_points: np.ndarray` – generated using `np.geomspace(0.001, 3, 100)`
* `sigma_code: np.ndarray` – cross section results loaded via `np.loadtxt(code_file)`
* `fit_params: list[tuple[float]]` – stores `p0` to `p3` from `curve_fit`
* `ratio_data[:, 0]` – mass column extracted as a NumPy slice

| Concept / Structure                       | Python Type Annotation               | Notes                                                          |
| ----------------------------------------- | ------------------------------------ | -------------------------------------------------------------- |
| Mass lists                                | `np.ndarray` or `list[float]`        | Used for generating and iterating over dark photon mass points |
| Cross section arrays                      | `np.ndarray`                         | Used throughout for storing computed and simulated values      |
| Polynomial coefficients (e.g. `p0`, `p1`) | `list[float]` or `tuple[float, ...]` | Used for regression modeling and correction factors            |
| Data tables (e.g. `code_data`)            | `np.ndarray`                         | From file I/O using `np.loadtxt()`                             |

---

## 3. Structured Configurations

### Examples

* `options = Options()` – class instance storing values like `AtomicNumber`, `Display`, `OutputDir`
* `fit_params[target_label].append((E0, popt, perr))` – stores coefficients and uncertainties per energy

| Concept           | Python Representation | Notes                                                                      |
| ----------------- | --------------------- | -------------------------------------------------------------------------- |
| Global Options    | `class Options:`      | Used to simulate command-line argument handling in notebooks or batch runs |
| Fitted Parameters | `List[Tuple[float]]`  | Used to store coefficients and uncertainties from regression fits          |

---

## 4. Summary

The project is written in Python with heavy use of:

* `numpy` for arrays and mathematical computation
* `scipy` for numerical integration and fitting
* `matplotlib` for plotting
* `tensorflow` optionally for fitting comparisons

The import of `ctypes` remains vestigial.
