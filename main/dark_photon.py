"""
This script computes dark photon cross sections for given targets and energies, compares them with MadGraph simulations,
performs ratio analyses, fits correction factors, and estimates yield enhancements and double ratios.
It saves outputs, plots graphs with uncertainty bands, fits polynomials, and derives final correction factor functions.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from optparse import OptionParser
from scipy.optimize import curve_fit
import pandas as pd
import ctypes  # Included for system-level compatibility or future extensions

# Check Python version
if sys.version_info[0] < 3:
    print('You need to run this with Python 3!')
    sys.exit(1)

# Constants
ALPHA = 1 / 137
M_E = 0.000510998  # GeV
M_P = 0.938272  # GeV
MU_P = 2.79
N_A = 6.022140857e23
SIGMA_UNITS = 1e-27

# Parser options
parser = OptionParser()
parser.add_option("-Z", "--AtomicNumber", type=float, default=74, help="Target Z (default: 74 for W)")
parser.add_option("-A", "--MassNumber", type=float, default=183.84, help="Target A (default: 183.84 for W)")
parser.add_option("-O", "--OutputDir", default="Output/", help="Output directory for saving results")
parser.add_option("-G", "--GitDir", default="MadGraphData/", help="GitHub directory containing MadGraph data")
parser.add_option("-P", "--PolyFit", type=int, default=3, help="Polynomial fit degree (3 or 4)")
parser.add_option("-S", "--SavePlots", action='store_true', default=False, help="Save plots to output directory")
parser.add_option("-D", "--Display", action='store_true', default=False, help="Display plots interactively")

(options, args) = parser.parse_args()

# Ensure output directories exist
os.makedirs(options.OutputDir, exist_ok=True)

# Define beam energies and target materials
targets = {'W': (74, 183.84), 'Al': (13, 26.98)}
energies = [4, 8, 12, 16, 20]


# --- Random Number Utility (used for synthetic data testing) ---
def generate_noisy_ratio_data(mass_points: np.ndarray, true_coeffs: list[float], noise_std: float = 0.05) -> tuple[
    np.ndarray, np.ndarray]:
    x = mass_points * 1000  # Convert to MeV scale as in fits
    y_true = true_coeffs[0] + true_coeffs[1] * x + true_coeffs[2] * x ** 2 + true_coeffs[3] * x ** 3
    noise = np.random.normal(0, noise_std, size=mass_points.shape)
    y_noisy = y_true + noise
    return y_true, y_noisy


# --- Comparison Sample Plot Using Synthetic Noisy Data ---
def compare_fit_with_noisy_data():
    mass_test = np.linspace(0.001, 0.1, 50)
    true_coeffs = [1.0, -0.01, 0.0002, -1e-6]
    y_true, y_noisy = generate_noisy_ratio_data(mass_test, true_coeffs)

    def cubic(x, a, b, c, d):
        return a + b * x + c * x ** 2 + d * x ** 3

    popt, _ = curve_fit(cubic, mass_test * 1000, y_noisy)
    y_fit = cubic(mass_test * 1000, *popt)

    plt.figure(figsize=(8, 5))
    plt.plot(mass_test * 1000, y_true, label="True Function", linestyle='--')
    plt.scatter(mass_test * 1000, y_noisy, label="Noisy Data", s=15)
    plt.plot(mass_test * 1000, y_fit, label="Fitted Function", color='red')
    plt.xlabel("$m_{A'}$ [MeV]")
    plt.ylabel("Synthetic Cross Section Ratio")
    plt.title("Synthetic Fit vs Noisy Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Run the synthetic comparison
compare_fit_with_noisy_data()

# --- BEGIN STAGE 1 TO 7 IMPLEMENTATION ---

# Stage 1: Compute cross sections and save to file
mass_points = np.array([
    0.010, 0.020, 0.030, 0.040, 0.050,
    0.060, 0.070, 0.080, 0.090
])  # GeV — matches simulation data

def compute_cross_section(mA, E0, Z, A, epsilon=1e-3):
    """
    Dark photon cross section using full WW approximation,
    including elastic and inelastic chi contributions.
    """
    if mA <= 0 or E0 <= mA:
        print(f"Warning: Unphysical inputs mA = {mA:.4f}, E0 = {E0:.4f}")
        return 0.0

    alpha = ALPHA
    me = M_E

    try:
        beta = np.sqrt(1 - (mA / E0) ** 2)
    except ValueError:
        print(f"Math domain error in beta for mA = {mA}, E0 = {E0}")
        return 0.0

    delta = max(mA / E0, me ** 2 / mA ** 2, (mA * me) / E0)
    if delta <= 0:
        print(f"Warning: delta is non-positive ({delta}) for mA = {mA}, E0 = {E0}")
        return 0.0

    log_term = np.log(1 / delta)
    if log_term <= 0:
        print(f"Warning: log(1/delta) <= 0 for delta = {delta}")
        return 0.0

    # Approximate chi_elastic and chi_inelastic
    try:
        chi_elastic = Z ** 2 * max(np.log((184.15 / Z ** (1 / 3)) ** 2 * E0 ** 2 / (mA ** 2)) - 1, 0.0)
        chi_inelastic = Z * max(np.log(1 + (1.0 * E0 / mA)) - 1, 0.0)
    except Exception as e:
        print(f"Chi computation failed: {e}")
        return 0.0

    chi = chi_elastic + chi_inelastic

    prefactor = (alpha ** 3 * epsilon ** 2 * beta * chi) / (mA ** 2)
    sigma = prefactor * log_term / A * 1e6 * SIGMA_UNITS

    if np.isnan(sigma) or sigma < 0:
        print(f"Invalid sigma value: {sigma} for mA = {mA}, E0 = {E0}")
        return 0.0

    return sigma

for target_label, (Z, A) in targets.items():
    for E0 in energies:
        filename = f"cross_section_code_{target_label}_{E0}GeV.txt"
        filepath = os.path.join(options.OutputDir, filename)
        with open(filepath, 'w') as f:
            f.write("# mA\tSigma_code\n")
            for mA in mass_points:
                sigma = compute_cross_section(mA, E0, Z, A)
                f.write(f"{mA:.6f}\t{sigma:.6e}\n")
        print(f"Saved computed cross sections to {filepath}")

# Stage 2: Load MadGraph results and plot comparisons
for target_label in targets:
    plt.figure(figsize=(10, 6))
    for E0 in energies:
        code_file = os.path.join(options.OutputDir, f"cross_section_code_{target_label}_{E0}GeV.txt")
        sim_file = os.path.join(options.GitDir, f"cross_section_sim_{target_label}_{E0}GeV.txt")

        try:
            code_data = np.loadtxt(code_file)
            sim_data = np.loadtxt(sim_file)
        except Exception as e:
            print(f"Missing data file for {target_label} at {E0} GeV: {e}")
            continue

        m_code, sigma_code = code_data[:, 0], code_data[:, 1]
        m_sim, sigma_sim, sigma_err = sim_data[:, 0], sim_data[:, 1], sim_data[:, 2]

        # Plot with uncertainties
        plt.plot(m_code * 1000, sigma_code, label=f"{E0} GeV (code)", linestyle='--')
        plt.errorbar(m_sim * 1000, sigma_sim, yerr=sigma_err, fmt='o', label=f"{E0} GeV (sim)")

    plt.xlabel("$m_{A'}$ [MeV]")
    plt.ylabel("Cross Section [$\mu$b]")
    plt.title(f"Cross Section Comparison - Target: {target_label}")
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()

    if options.SavePlots:
        plot_path = os.path.join(options.OutputDir, f"CrossSectionComparison_{target_label}.pdf")
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
    if options.Display:
        plt.show()
    else:
        plt.close()

# Stage 3: Compute cross section ratios and propagate uncertainties
for target_label in targets:
    for E0 in energies:
        code_file = os.path.join(options.OutputDir, f"cross_section_code_{target_label}_{E0}GeV.txt")
        sim_file = os.path.join(options.GitDir, f"cross_section_sim_{target_label}_{E0}GeV.txt")
        ratio_file = os.path.join(options.OutputDir, f"cross_section_ratio_{target_label}_{E0}GeV.txt")

        try:
            code_data = np.loadtxt(code_file)
            sim_data = np.loadtxt(sim_file)
        except Exception as e:
            print(f"Skipping ratio computation for {target_label}, {E0} GeV due to missing file: {e}")
            continue

        m_sim, sigma_sim, sigma_err = sim_data[:, 0], sim_data[:, 1], sim_data[:, 2]
        m_code, sigma_code = code_data[:, 0], code_data[:, 1]

        ratio = sigma_sim / sigma_code
        ratio_err = ratio * np.sqrt((sigma_err / sigma_sim) ** 2)  # propagate error only from sim

        np.savetxt(ratio_file, np.column_stack((m_sim, ratio, ratio_err)),
                   header="mA\tRatio\tError", fmt="%.6f\t%.6e\t%.6e")
        print(f"Saved ratio data to {ratio_file}")

# Stage 4: Fit cubic polynomials to the ratios and extract parameters
fit_params = {target_label: [] for target_label in targets}

for target_label in targets:
    plt.figure(figsize=(10, 6))
    for E0 in energies:
        ratio_file = os.path.join(options.OutputDir, f"cross_section_ratio_{target_label}_{E0}GeV.txt")
        try:
            ratio_data = np.loadtxt(ratio_file)
        except Exception as e:
            print(f"Skipping fit for {target_label}, {E0} GeV due to missing ratio file: {e}")
            continue

        m_vals, ratio_vals, ratio_errs = ratio_data[:, 0], ratio_data[:, 1], ratio_data[:, 2]


        def cubic(x, a, b, c, d):
            return a + b * x + c * x ** 2 + d * x ** 3


        try:
            popt, pcov = curve_fit(cubic, m_vals * 1000, ratio_vals, sigma=ratio_errs, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            fit_params[target_label].append((E0, popt, perr))

            xfit = np.linspace(min(m_vals), max(m_vals), 300) * 1000
            yfit = cubic(xfit, *popt)

            plt.errorbar(m_vals * 1000, ratio_vals, yerr=ratio_errs, fmt='o', label=f"{E0} GeV data")
            plt.plot(xfit, yfit, label=f"{E0} GeV fit")

        except RuntimeError as e:
            print(f"Fit failed for {target_label} at {E0} GeV: {e}")

    plt.xlabel("$m_{A'}$ [MeV]")
    plt.ylabel("Cross Section Ratio (Sim/Code)")
    plt.title(f"Ratio Polynomial Fits - Target: {target_label}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    if options.SavePlots:
        plt.savefig(os.path.join(options.OutputDir, f"RatioFit_{target_label}.pdf"))
    if options.Display:
        plt.show()
    else:
        plt.close()

# --- Stage 5: Fit parameters vs. energy and construct correction function ---
param_labels = ["p0", "p1", "p2", "p3"]
for target_label in targets:
    params_by_energy = {label: [] for label in param_labels}
    errors_by_energy = {label: [] for label in param_labels}
    energies_recorded = []

    for (E0, coeffs, errors) in fit_params[target_label]:
        energies_recorded.append(E0)
        for i, label in enumerate(param_labels):
            params_by_energy[label].append(coeffs[i])
            errors_by_energy[label].append(errors[i])

    print(f"\nCorrection Function for target {target_label}:")
    for i, label in enumerate(param_labels):
        x = np.array(energies_recorded)
        y = np.array(params_by_energy[label])
        yerr = np.array(errors_by_energy[label])


        def quadratic(E, a, b, c):
            return a * E ** 2 + b * E + c


        try:
            popt, pcov = curve_fit(quadratic, x, y, sigma=yerr, absolute_sigma=True)
            a, b, c = popt
            print(f"{label}(E) = ({a:.3e})E^2 + ({b:.3e})E + ({c:.3e})")

            Efit = np.linspace(min(x), max(x), 100)
            yfit = quadratic(Efit, *popt)

            plt.figure()
            plt.errorbar(x, y, yerr=yerr, fmt='o', label=f"{label} data")
            plt.plot(Efit, yfit, label=f"Quadratic fit")
            plt.title(f"{label} vs Beam Energy - {target_label}")
            plt.xlabel("Beam Energy [GeV]")
            plt.ylabel(label)
            plt.legend()
            plt.grid(True)
            if options.SavePlots:
                plt.savefig(os.path.join(options.OutputDir, f"{target_label}_{label}_fit.pdf"))
            if options.Display:
                plt.show()
            else:
                plt.close()

        except RuntimeError as e:
            print(
                f"Fit error for {label} ({target_label}): {e}")

# Stage 6: Yield enhancement and baseline comparison (relative to Tungsten 4 GeV)
yield_data = {t: {E: None for E in energies} for t in targets}

for target_label, (Z, A) in targets.items():
    for E0 in energies:
        file_path = os.path.join(options.OutputDir, f"cross_section_code_{target_label}_{E0}GeV.txt")
        try:
            data = np.loadtxt(file_path)
            yield_data[target_label][E0] = data
        except Exception as e:
            print(f"Yield load failed for {target_label}, {E0} GeV: {e}")

baseline = yield_data['W'][4]  # baseline is Tungsten at 4 GeV

for target_label in targets:
    plt.figure(figsize=(10, 6))
    for E0 in energies:
        try:
            current = yield_data[target_label][E0][:, 1]
            baseline_vals = baseline[:, 1]
            mass_vals = baseline[:, 0]
            yield_enhancement = current / baseline_vals

            plt.plot(mass_vals * 1000, yield_enhancement, label=f"{E0} GeV")
            np.savetxt(os.path.join(options.OutputDir, f"yield_enh_{target_label}_{E0}GeV.txt"),
                       np.column_stack((mass_vals, yield_enhancement)), fmt="%.6f\t%.6e",
                       header="mA\tYieldEnhancement")
        except Exception as e:
            print(f"Yield enhancement failed for {target_label}, {E0} GeV: {e}")

    plt.title(f"Yield Enhancement vs $m_{{A'}}$ - {target_label}")
    plt.xlabel("$m_{A'}$ [MeV]")
    plt.ylabel("Yield Enhancement")
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    if options.SavePlots:
        plt.savefig(os.path.join(options.OutputDir, f"YieldEnh_{target_label}.pdf"))
    if options.Display:
        plt.show()
    else:
        plt.close()

# Stage 7: Double Ratio and Quartic Correction Factor
quartic_params = {target: [] for target in targets}

yield_enh_files = {
    'Al': {
        4:  "Al_4 GeV_MadGraph_Yield Enhancements.txt",
        8:  "Al_8 GeV_MadGraph_Yield Enhancements.txt",
        12: "Al_12 GeV_MadGraph_Yield Enhancements.txt",
        16: "Al_16 GeV_MadGraph_Yield Enhancements.txt",
        20: "Al_20 GeV_MadGraph_Yield Enhancements.txt"
    },
    'W': {
        4:  "W_4 GeV_MadGraph_Yield Enhancements.txt",
        8:  "W_8 GeV_MadGraph_Yield Enhancements.txt",
        12: "W_12 GeV_MadGraph_Yield Enhancements.txt",
        16: "W_16 GeV_MadGraph_Yield Enhancements.txt",
        20: "W_20 GeV_MadGraph_Yield Enhancements.txt"
    }
}
file_sim = yield_enh_files[target_label][E0]

for target_label in targets:
    for E0 in energies:
        try:
            file_sim = yield_enh_files[target_label][E0]
            data_code = np.loadtxt(file_code)
            data_sim = np.loadtxt(file_sim)
            m_vals = data_code[:, 0]
            enh_code = data_code[:, 1]
            enh_sim = data_sim[:, 1]
            err_sim = data_sim[:, 2]

            double_ratio = enh_sim / enh_code
            double_ratio_err = double_ratio * (err_sim / enh_sim)

            save_path = os.path.join(options.OutputDir, f"double_ratio_{target_label}_{E0}GeV.txt")
            np.savetxt(save_path, np.column_stack((m_vals, double_ratio, double_ratio_err)),
                       header="mA\tDoubleRatio\tError", fmt="%.6f\t%.6e\t%.6e")


            def quartic(x, a, b, c, d, e):
                return a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4


            popt, pcov = curve_fit(quartic, m_vals * 1000, double_ratio, sigma=double_ratio_err, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            quartic_params[target_label].append((E0, popt, perr))

            xfit = np.linspace(min(m_vals), max(m_vals), 300) * 1000
            yfit = quartic(xfit, *popt)

            plt.figure()
            plt.errorbar(m_vals * 1000, double_ratio, yerr=double_ratio_err, fmt='o', label=f"{E0} GeV")
            plt.plot(xfit, yfit, label="Quartic Fit")
            plt.title(f"Double Ratio vs Mass - {target_label}, {E0} GeV")
            plt.xlabel("$m_{A'}$ [MeV]")
            plt.ylabel("Double Ratio")
            plt.grid(True)
            plt.legend()
            if options.SavePlots:
                plt.savefig(os.path.join(options.OutputDir, f"DoubleRatioFit_{target_label}_{E0}GeV.pdf"))
            if options.Display:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Double ratio failed for {target_label}, {E0} GeV: {e}")

# Fit quartic parameters p0–p4 as cubic in energy and report correction formula
for target_label in targets:
    print(f"\nQuartic Correction Factor for target {target_label}:")
    for i, label in enumerate(["p0", "p1", "p2", "p3", "p4"]):
        energies_recorded = []
        param_vals = []
        param_errs = []
        for (E0, coeffs, errs) in quartic_params[target_label]:
            energies_recorded.append(E0)
            param_vals.append(coeffs[i])
            param_errs.append(errs[i])


        def cubic(E, a, b, c, d):
            return a * E ** 3 + b * E ** 2 + c * E + d


        try:
            popt, _ = curve_fit(cubic, energies_recorded, param_vals, sigma=param_errs, absolute_sigma=True)
            print(f"{label}(E) = ({popt[0]:.3e})E^3 + ({popt[1]:.3e})E^2 + ({popt[2]:.3e})E + ({popt[3]:.3e})")
        except Exception as e:
            print(f"Fit error for {label} ({target_label}): {e}")
