# Results

The analytical and simulation-based models for dark photon production cross sections were compared using yield enhancements and cross section ratios. These comparisons were used to derive correction factors that align the semi-analytical model with detailed Geant4/MadGraph simulations.

## 1. Cross Section Comparison

Initial comparisons revealed that the Python-based analytical model overestimates cross section values, especially at higher dark photon masses $\( m_{A'} \)$.  
Figure 1 shows this discrepancy for both tungsten and aluminum targets, where simulation and code results diverge significantly at high $\( m_{A'} \)$.

![image](https://github.com/user-attachments/assets/df500a8a-2168-4b6e-845a-441dd57fec5e)
![CrossSectionsAluminiumFinal](https://github.com/user-attachments/assets/62d4cae2-7b66-4c5d-8242-3185bf52ef26)
![image](https://github.com/user-attachments/assets/5963456b-bdfd-4ad9-964f-d8deb038494f)
![image](https://github.com/user-attachments/assets/5324f8da-3495-4f7e-9e62-d6fc0bc11579)

Figure 1: Graphs of Cross Section values and ratio of Cross Sections against dark photon masses for
Tungsten and Aluminum.

---

## 2. Yield Enhancement $\( \eta \)$
The yield enhancement $\eta$ is defined as:

$$
\eta_{X,E} = \frac{\sigma_{X,E}}{\sigma_{W,4\,\text{GeV}}}
$$

Where:

* $X \in \{\text{Aluminium(Al)}, \text{Tungsten(W)}\}$ is the target
* $E \in \{4, 8, 12, 16, 20\}$ GeV is the beam energy
* $\sigma_{W,4}$ is the baseline cross section for tungsten at 4 GeV

This normalization enables a relative comparison across energies and materials.  
Plots of $\( \eta \)$ versus $\( m_{A'} \)$ (Figure 2) show increasing yields with higher beam energy and mass, and demonstrate that code and simulation results follow similar patterns.

![image](https://github.com/user-attachments/assets/5b7fe826-3471-4b46-a59d-f56d2390360c)
![image](https://github.com/user-attachments/assets/ac3a06e8-0cff-4408-b1c7-73c4382b37c9)
Figure 2: Graphs of Yield Enhancement for Tungsten and Aluminum.



---

## 3. Double Ratio $\( \zeta \)$ and Correction Factor $\( f_\eta \)$

To quantify the mismatch between analytical and simulation results, the **double ratio** is introduced:

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

where 

$$
\begin{aligned}
p_0 &= (0.00419161)(E_0)^3 + (-0.151256)(E_0)^2 + (1.46758)E_0 + (-2.66196) \\
p_1 &= (-1.20314 \times 10^{-5})(E_0)^3 + (0.000414147)(E_0)^2 + (-0.00346943)E_0 + (0.00802469) \\
p_2 &= (1.094 \times 10^{-8})(E_0)^3 + (-3.51 \times 10^{-7})(E_0)^2 + (2.39048 \times 10^{-6})E_0 + (-4.62571 \times 10^{-6}) \\
p_3 &= (-3.309 \times 10^{-12})(E_0)^3 + (8.9315 \times 10^{-11})(E_0)^2 + (-2.44723 \times 10^{-10})E_0 + (-2.3859 \times 10^{-10}) \\
p_4 &= (2.542 \times 10^{-16})(E_0)^3 + (-3.774 \times 10^{-15})(E_0)^2 + (-5.42588 \times 10^{-14})E_0 + (2.61163 \times 10^{-13})
\end{aligned}
$$

This yields a 2D correction factor $f_\eta(m_{A'}, E)$ that adjusts the code-based yield predictions to match simulations.

Figure 3 shows the double ratio fits:

![image](https://github.com/user-attachments/assets/79fc7997-33d1-4167-a531-d5d336acc546)
![image](https://github.com/user-attachments/assets/eae0a1c3-2736-4c2a-910a-a0a4161d5296)

Figure 3: Graphs of Double Ratios for Tungsten and Aluminum.

Figure 4 shows a quartic polynomial fit applied to the Tungsten double ratio plots.

![image](https://github.com/user-attachments/assets/6225f021-e24b-470d-966f-e54308f4486e)
Figure 4: Degree 4 polynomial fits applied to the Double Ratio plots for Tungsten. The fits almost perfectly capture the evolution
of graphs.

Figure 5 plots the energy dependence of the polynomial coefficients.
![image](https://github.com/user-attachments/assets/2d0a7d4d-a51c-402b-b386-f1e36d1a205c)
![image](https://github.com/user-attachments/assets/3985f234-ec10-4c49-9b83-6c7f14a92160)
![image](https://github.com/user-attachments/assets/ec48bec6-6323-4122-8fd4-5ead66753b66)
![image](https://github.com/user-attachments/assets/a2bd4869-a4be-4878-abc8-45015d7ffb21)
![image](https://github.com/user-attachments/assets/d9ef039e-a0f2-4eb8-8660-b980a6eb7b3e)

Figure 5: The parameters are plotted as functions of electron beam energy. Degree 3 polynomial fits are applied to obtain an analytical relationship for the evolution of parameters with energy.

Validation of this correction (Figure 6) shows that applying $\( f_\eta \)$ causes the normalized ratio $\( \zeta / f_\eta \)$ to converge to 1, confirming its accuracy for tungsten targets.

![image](https://github.com/user-attachments/assets/88ad507b-c0b3-4046-a5e4-a44c32d274da)
Figure 6: Curves for all energies are reasonably close to 1, indicating that the function f is an appropriate correction factor


> **Note:** Parameterization for aluminum targets was attempted (Figure 7), but fits were not robust at high $\( m_{A'} \)$. This remains a subject for future work.

![image](https://github.com/user-attachments/assets/95874beb-4131-4185-9471-efc84c491f68)

Figure 7: Degree 4 fits applied to Aluminium curves do not capture their evolution precisely enough, especially at higher mass
values.

---

## 4. Cross Section Ratio \( \kappa \) and Correction Factor \( f_\kappa \) (Equations 5, 6, 7; Figures 10–13)

The **cross section ratio** compares total cross sections from simulation and code:

$$
\kappa = \frac{\sigma_{X,E}^{\text{sim}}}{\sigma_{X,E}^{\text{code}}}
$$

This ratio is modeled by a cubic polynomial:

$$
f_\kappa(m_{A'}, E) = p_0 + p_1 (1000\,m_{A'}) + p_2 (1000\,m_{A'})^2 + p_3 (1000\,m_{A'})^3
\tag{6}
$$

Each \( p_i \) is a quadratic function of energy:

$$
p_i(E) = \alpha_i E^2 + \beta_i E + \gamma_i
\tag{7}
$$

Figure 10 shows the fitted curves for tungsten, and Figure 11 shows the energy dependence of the polynomial coefficients.  
The resulting correction factor \( f_\kappa \) accurately predicts simulation cross sections when applied to the code-based model, as shown in Figure 12.

> **Note:** As with \( f_\eta \), aluminum fits (Figure 13) were unstable, and future work is needed to build accurate correction factors for lighter target materials.

---

## Summary

- **Correction factors \( f_\eta \) and \( f_\kappa \)** successfully map analytical predictions to simulation results for tungsten.
- **Quartic and cubic fits** are validated against numerical data and exhibit strong convergence to unity when applied.
- **Limitations for aluminum** remain due to poor fit quality at higher dark photon masses.

These correction models enable **fast and accurate cross section predictions** across a broad parameter space, significantly improving the practical utility of the semi-analytical approach in upcoming LDMX studies.
