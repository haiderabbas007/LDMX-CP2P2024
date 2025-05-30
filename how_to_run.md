# How to Run the Dark Photon Analysis Code

This document provides step-by-step instructions for setting up the environment, running the scripts, and understanding the outputs for the Dark Photon cross section and correction factor analysis.

---

## Requirements

Make sure you have the following installed:

* Python 3.10+
* `pip`
* Git
* Optional: [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (for Windows users)

---

## 1. Clone the Repository

```bash
git clone https://github.com/haiderabbas007/LDMX-CP2P2024.git
cd LDMX-CP2P2024
```

---

## 2. Install Dependencies

### Using pip (recommended for quick start):

```bash
pip install -r requirements.txt
```

Alternatively, manually install:

```bash
pip install numpy matplotlib scipy pandas tensorflow flake8
```

---

## 3. Run the Main Analysis Script

The main analysis script computes cross sections, compares with MadGraph simulations, and generates correction factors and plots.

```bash
python dark_photon_cross_section.py --SavePlots --Display
```

Optional arguments:

* `--AtomicNumber`: Target Z (default: 74)
* `--MassNumber`: Target A (default: 183.84)
* `--OutputDir`: Output directory (default: Output/)
* `--GitDir`: Directory for MadGraph data (default: MadGraphData/)
* `--PolyFit`: Degree of polynomial for ratio fit (default: 3)

---

## 4. Use TensorFlow Fit Model

To use the TensorFlow-based curve fitting:

```bash
python tensorflow.py
```

This will generate:

* `synthetic_ratio_data_tf.txt`: Fitted values
* `synthetic_ratio_plot_comparison.png` and `.pdf`: Comparison plots

---

## 5. Run Unit Tests and Lint Checks

### Run all tests:

```bash
python -m unittest discover -s tests -p 'test*.py'
```

### Run linting:

```bash
flake8 . --count --show-source --statistics --max-line-length=120
```

---

## 6. Use Docker + Devcontainer

Ensure you have Docker and VS Code installed with the Dev Containers extension.

* The `.devcontainer` folder contains setup for running inside a containerized development environment.
* Launch VS Code and reopen the project in container when prompted.

---

## Outputs

All results, fits, and plots are saved in the directory specified by `--OutputDir`. This includes:

* Cross section tables
* Ratio fits
* Polynomial correction plots
* Yield enhancement and double ratio files

---

## Tips

* Set a seed using `np.random.seed(42)` to get reproducible results.
* Run in WSL or a Linux environment for best compatibility.

---

## Troubleshooting

* If `tensorflow` fails to install on Windows, use WSL or install inside a virtual environment.
* Check paths to MadGraph data to ensure file availability.
* Use `--Display` for interactive plotting if running locally.
