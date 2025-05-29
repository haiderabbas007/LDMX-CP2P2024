# Light Dark Matter Experiment

## Introduction

The Light Dark Matter eXperiment (LDMX) [1] is a proposed small-scale accelerator experiment with broad sensitivity to light dark matter (LDM) and mediator particles in the sub-GeV mass range. LDMX is designed to use missing momentum and energy techniques in multi-GeV electron fixed-target collisions to probe extremely weak couplings to electrons, extending into regions motivated by thermal freeze-out dark matter scenarios.

Figure 1 shows the experimental setup:

![image](https://github.com/user-attachments/assets/92be5c85-05cc-4081-ac47-0ed965b2063e)
Figure 1: Detector design and fixed-target setup being impinged by a high energy electron beam [1].


Traditional direct detection experiments have primarily targeted weakly interacting massive particles (WIMPs) with masses in the GeV–TeV range. However, stringent null results from these experiments have pushed researchers to explore lighter alternatives. LDMX addresses the MeV–GeV range, which remains challenging for conventional detectors but is well-motivated in “hidden sector” models. In these theories, dark matter interacts via a new force mediated by a dark photon, and is neutral under all Standard Model forces.

By using a high-luminosity electron beam directed at a thin target (e.g., tungsten or aluminum), LDMX aims to identify events where a dark photon is produced via bremsstrahlung and decays into invisible particles. The signature is a significant energy-momentum imbalance in the recoiling electron. This makes LDMX uniquely capable of identifying new weakly coupled forces and particles in previously unexplored regions of parameter space.


This project compares dark photon cross sections computed using:

* A semi-analytical Python-based implementation of the Weizsäcker-Williams approximation [2].
* Full Geant4/MadGraph simulations produced by the LDMX collaboration.

The objective is to refine the analytical model using correction factors that align it with simulation data, improving both computational speed and predictive accuracy.

## Theoretical Motivation

Dark matter may interact via a hidden sector mediated by a dark photon, a hypothetical gauge boson that couples to the Standard Model via kinetic mixing. These dark photons can be radiated off electrons in fixed-target collisions and may decay invisibly to light dark matter particles.

Two types of processes are of interest:

* Direct production of dark matter: e + N → e + χ̅ + χ
* Mediated production via dark photon: e + N → e + A′; A′ → χ̅ + χ

Figure 2 shows the relevant Feynman Diagrams for these processes:
![image](https://github.com/user-attachments/assets/dbe9de21-f4f2-4441-8917-58a9b7f094d6)
Figure 2: Left panel: Feynman diagram for direct dark matter particle-antiparticle production. Right panel: Feynman diagram for radiation of a mediator particle off a beam electron, followed by its decay into dark matter particles. Measuring both of these (and similar) reactions is the primary science goal of LDMX and will provide broad and powerful sensitivity to light dark matter and many other types of dark sector physics [1].


LDMX aims to detect these by reconstructing the recoil electron and identifying events with large missing energy and momentum.

## Cross Section Computation

The analytical expression for the dark photon production cross section $\sigma$ is derived from the Weizsäcker-Williams approximation:

$$
\sigma \propto \frac{\alpha^3 \epsilon^2 \beta}{m_{A'}^2} \cdot \chi \cdot \left[\log\left(\delta^{-1}\right) + \mathcal{O}(1)\right]
$$

Where:

- $\epsilon$ is the kinetic mixing parameter  
- $m_{A'}$ is the dark photon mass  
- $\beta = \sqrt{1 - (m_{A'}/E_0)^2}$  
- $\delta \sim \max\left(m_{A'}/E_0, \, m_e^2/m_{A'}^2, \, m_e/E_0\right)$  
- $\chi = \chi_{\text{elastic}} + \chi_{\text{inelastic}}$

The $\chi$ integrals are computed numerically over elastic and inelastic form factors:


### Elastic $\chi_{\text{elastic}}$ Integral:

$$
\chi_{\text{elastic}} = \int_{t_{\min}}^{t_{\max}} \frac{F_{\text{el}}^2(t)}{t} \, dt
$$

with:

$$
F_{\text{el}}(t) = \frac{Z}{1 + t / t_a}, \quad t_a \approx \left(111 Z^{-1/3}\right)^{-2}
$$


### Inelastic $\chi_{\text{inelastic}}$ Integral:

$$
\chi_{\text{inelastic}} = \int_{t_{\min}}^{t_{\max}} \frac{F_{\text{in}}^2(t)}{t} \, dt
$$

with:

$$
F_{\text{in}}(t) = \frac{Z}{1 + t / t_b}, \quad t_b \approx \left(773 Z^{-2/3}\right)^{-2}
$$



## Yield Enhancement

The yield enhancement $\eta$ is defined as:

$$
\eta_{X,E} = \frac{\sigma_{X,E}}{\sigma_{W,4\,\text{GeV}}}
$$

Where:

* $X \in \{\text{Aluminium(Al)}, \text{Tungsten(W)}\}$ is the target
* $E \in \{4, 8, 12, 16, 20\}$ GeV is the beam energy
* $\sigma_{W,4}$ is the baseline cross section for tungsten at 4 GeV

## Double Ratio and Correction Factor

To compare model vs simulation:

$$
\zeta = \frac{\eta^{\text{sim}}}{\eta^{\text{code}}}
$$

This double ratio is then fitted to a quartic polynomial in dark photon mass:

$$
f_\eta(m_{A'}, E) = p_0 + p_1 (1000 m_{A'}) + p_2 (1000 m_{A'})^2 + p_3 (1000 m_{A'})^3 + p_4 (1000 m_{A'})^4
$$

Each coefficient $p_i$ is then parameterized as a cubic function of energy:

$$
p_i(E) = a_i E^3 + b_i E^2 + c_i E + d_i
$$

This yields a 2D correction factor $f_\eta(m_{A'}, E)$ that adjusts the code-based yield predictions to match simulations.

## Cross Section Ratio and Cubic Fit

Similarly, the ratio of cross sections $\kappa$ is defined as:

$$\kappa = \frac{\sigma_{X, E}^{\textrm{sim}}}{\sigma_{X, E}^{\textrm{code}}}$$

This ratio is fitted to a cubic polynomial:

$$
f_\kappa(m_{A'}, E) = p_0 + p_1 (1000\,m_{A'}) + p_2 (1000\,m_{A'})^2 + p_3 (1000\,m_{A'})^3
$$

Each $p_i$ is parameterized as a quadratic function of energy to construct $f_\kappa(m_{A'}, E)$.

## Summary of Results

* The Python model systematically overestimates the cross section at high $m_{A'}$.
* The correction factors $f_\eta$ (quartic) and $f_\kappa$ (cubic) accurately align the model to simulation.
* Final correction functions allow efficient analytical estimation of realistic cross sections and yields.

## References
- [1] Åkesson, T. et al. (2018). “Light Dark Matter eXperiment (LDMX).” [arXiv:1808.05219](https://arxiv.org/abs/1808.05219)
- [2] Bjorken, D. et al. (2009). “New Fixed-Target Experiments to Search for Dark Gauge Forces.” [arXiv:0906.0580](https://arxiv.org/abs/0906.0580)
- [3] Geoffrey Mullier, “Dark Photon Cross Section Calculator,” [CERN GitLab](https://gitlab.cern.ch/gmullier/dark-photon-cross-section-calculator)
