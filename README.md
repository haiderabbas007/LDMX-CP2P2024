# README.md

## Project: Dark Photon Cross Section and Correction Factor Analysis

This repository contains scripts and data for calculating, comparing, and visualizing dark photon cross sections. It also performs polynomial fits, correction factor derivations, and TensorFlow-based model comparisons.

---

## Directory Structure

```
.
├── .devcontainer/           # VSCode development container support
│   ├── Dockerfile
│   └── devcontainer.json
├── .github/workflows/      # GitHub Actions CI
│   └── lint-and-test.yml
├── data/                   # Processed and raw data files
│   ├── raw-data-combined/
│   └── raw-data-text/
├── main/                   # Main analysis scripts
│   ├── dark_photon.py
│   ├── data_types.md
│   ├── introduction_to_physics.md
│   └── results.md
├── random-numbers/         # Synthetic data generator
│   ├── random_numbers.py
│   └── random_numbers.md
├── tensorflow/             # TensorFlow comparison model
│   ├── tensorflow.py
│   └── tensorflow.md
├── unit-tests/             # Unit tests for the codebase
│   ├── test_dark_photon.py
│   ├── test_random_numbers.py
│   └── test_tensorflow.py
├── requirements.txt        # Python dependencies
├── how_to_run.md           # Usage instructions
├── .gitignore              # Ignored files for version control
├── LICENSE                 # License information
└── README.md               # Project overview
```

---

## Requirements

* Python 3.10+
* pip
* Git
* (Optional) Docker + VS Code (with Dev Containers extension)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## What It Does

* Computes dark photon cross sections for various targets and beam energies
* Compares results with MadGraph simulations
* Performs ratio and polynomial fits (cubic/quartic)
* Derives correction functions vs energy
* Implements a TensorFlow-based fit model for noisy data
* Generates diagnostic plots and saves all data

---

##  How to Run

### Main Analysis

```bash
python main/dark_photon.py --SavePlots --Display
```

Optional flags:

* `--AtomicNumber` (Z): e.g., 13 (Al) or 74 (W)
* `--MassNumber` (A): e.g., 26.98 (Al) or 183.84 (W)
* `--PolyFit`: Degree of polynomial fit (3 or 4)
* `--OutputDir`: Directory to save results
* `--GitDir`: Location of MadGraph simulation data

### TensorFlow Model

```bash
python tensorflow/tensorflow.py
```

Outputs:

* `synthetic_ratio_data_tf.txt`
* Comparison plots: `.png` and `.pdf`

---

## Testing & Linting

Run tests:

```bash
python -m unittest discover -s unit-tests -p 'test*.py'
```

Run flake8 linting:

```bash
flake8 . --count --show-source --statistics --max-line-length=120
```

---

## Dev Container Support

This repo includes a `.devcontainer/` folder to support VS Code development inside a Docker container.

To use:

1. Install Docker and VS Code
2. Install the Dev Containers extension
3. Open the repo in VS Code → Reopen in Container

---

## Outputs

Results are saved in the specified `--OutputDir`, including:

* Cross section `.txt` files
* Ratio and double ratio tables
* Polynomial fit coefficients
* Correction factor formulas and plots

---

## Troubleshooting

* **TensorFlow install issue on Windows?** → Use WSL or Dev Container
* **Missing MadGraph files?** → Ensure files are in the right directory
* **Plots not displaying?** → Add `--Display` or check file saves

---
