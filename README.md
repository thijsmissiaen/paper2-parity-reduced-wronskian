# Parity-Reduced Transverse Wronskians, Jensen–Laguerre Moment Identities, and Division-Free Interval Enclosures for Riemann's $\Xi$-Function

**DOI (Zenodo / CERN):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23641698.svg)](https://doi.org/10.5281/zenodo.23641698) [`10.5281/zenodo.23641698`](https://doi.org/10.5281/zenodo.23641698)  
**Author:** Thijs Missiaen (Independent Researcher, Belgium · [ORCID: 0009-0007-3678-0810](https://orcid.org/0009-0007-3678-0810) · `thijs.missiaen@gmail.com`)  
**Series:** Transverse Wronskian of Riemann's $\Xi$-Function (Paper II · [Paper I DOI: 10.5281/zenodo.23637324](https://doi.org/10.5281/zenodo.23637324))  
**Repository:** [`https://github.com/thijsmissiaen/paper2-parity-reduced-wronskian`](https://github.com/thijsmissiaen/paper2-parity-reduced-wronskian)  
**MSC 2020:** Primary `11M06`, `65G30`; Secondary `30D15`, `26D15`, `11Y35`  
**Manuscript:** [`paper/paper2_parity_reduced_wronskian.pdf`](paper/paper2_parity_reduced_wronskian.pdf) (13 pages, AMS-LaTeX / pdfTeX)

---

> [!NOTE]
> **Scope & Prior-Art Context:** Because $\Xi(t)$ is real on the critical line $d = 0$, the raw transverse Wronskian $W(d,t) = \frac{1}{2}\partial_d\lvert\Xi(t-\mathrm{i}d)\rvert^2$ satisfies $W(0,t) \equiv 0$, creating a kinematic $0/0$ degeneracy in interval boxes touching $d = 0$. Dividing analytically by $d$ yields $\widetilde{W}(d,t) = W(d,t)/d$, whose Maclaurin coefficients $c _ {2m-1}(t) = m L _ m(\Xi;t)$ recover the classical differential invariants of **Jensen (1913)** and **Csordas–Varga (1990)**, with $\widetilde{W}(0,t) = L _ 1(\Xi;t) = (\Xi'(t))^2 - \Xi(t)\Xi''(t)$. Since RH is already verified up to $T = 3 \cdot 10^{12}$ (**Gram 1903**, **Titchmarsh–Comrie 1936**, **Platt–Trudgian 2021**), all interval enclosures on $t \le 55$ serve strictly as methodological benchmarks alongside a quantitative Higham $\lvert x\rvert^\top\lvert y\rvert$ roundoff audit showing why IEEE-754 `float64` quadrature breaks down exponentially ($\propto u _ {\mathrm{mach}}\mathrm{e}^{+\pi t/4}$).

## Abstract & Main Identities

Let $\xi(s) = \frac{1}{2}s(s-1)\pi^{-s/2}\Gamma(s/2)\zeta(s)$ be Riemann's completed zeta function with $s = \frac{1}{2} + d + \mathrm{i}t$ ($d = \sigma - \frac{1}{2}$) and $\Xi(t-\mathrm{i}d) := \xi(s) = R(d,t) + \mathrm{i}I _ {\mathrm{im}}(d,t)$. In terms of the canonical Wigner correlation kernel $I(y,t) = \int _ {-\infty}^\infty \Phi(\frac{x+y}{2})\Phi(\frac{x-y}{2})\cos(2tx)\,\mathrm{d}x = 2\pi\mathcal{W} _ \Phi(y/2, 2t)$ established in Paper I, the transverse Wronskian satisfies

$$W(d,t) := \frac{1}{2}\frac{\partial}{\partial d}\lvert\Xi(t-\mathrm{i}d)\rvert^2 = R \partial _ t I _ {\mathrm{im}} - I _ {\mathrm{im}} \partial _ t R = 16\int_0^\infty y\sinh(2d y) I(y,t)\,\mathrm{d}y.$$

1. **Canonical Parity-Reduced Wronskian and Classical Laguerre Boundary Value (Proposition 2.1):**
   By Schwarz reflection $\Xi(t+\mathrm{i}d) = \overline{\Xi(t-\mathrm{i}d)}$, $W(-d,t) = -W(d,t)$ with $\partial_d^{2k}W(0,t) \equiv 0$. Factoring $d$ via $\mathrm{sinc} _ {\mathrm{hyp}}(w) := \frac{\sinh w}{w} = \int_0^1 \cosh(sw)\,\mathrm{d}s \ge 1$ gives

$$\widetilde{W}(d,t) := \frac{W(d,t)}{d} = 32\int_0^\infty y^2 \mathrm{sinc} _ {\mathrm{hyp}}(2d y) I(y,t)\,\mathrm{d}y, \qquad \partial_d\widetilde{W}(0,t) \equiv 0,$$

   with boundary value equal to the classical Laguerre expression $\widetilde{W}(0,t) = L _ 1(\Xi;t) = (\Xi'(t))^2 - \Xi(t)\Xi''(t)$.

2. **Jensen–Csordas–Varga Moment Hierarchy and Local Multiplicity Jet (Theorem 3.1 & Proposition 3.2):**
   Expanding $\widetilde{W}(d,t) = \sum _ {m=1}^\infty c _ {2m-1}(t) d^{2m-2}$ links the even canonical Wigner moments to the classical Jensen–Laguerre invariants $L _ m(\Xi;t)$:

$$c _ {2m-1}(t) = \frac{2^{2m+3}}{(2m-1)!}\int_0^\infty y^{2m}I(y,t)\,\mathrm{d}y = m L _ m(\Xi;t) := m\sum _ {j=0}^{2m}\frac{(-1)^{m+j}}{j! (2m-j)!}\Xi^{(j)}(t)\Xi^{(2m-j)}(t).$$

   At any $m$-fold real zero $t_0 \in \mathbb{R}$, $c _ 1(t_0) = \dots = c _ {2m-3}(t_0) = 0$ and $c _ {2m-1}(t_0) = \frac{m}{(m!)^2}(\Xi^{(m)}(t_0))^2 > 0$. Using $\eta _ {\mathrm{hyp}}(w) := \frac{w\cosh w - \sinh w}{w^3} = \int_0^1\int_0^1 s^2\cosh(s\tau w)\,\mathrm{d}\tau\,\mathrm{d}s \ge \frac{1}{3}$, all mixed $(d,t)$ Taylor remainders of the non-harmonic auxiliaries in $\nabla\widetilde{W}$ are bounded directly by holomorphic strip suprema of $\Xi^{(m)}$ without assuming harmonicity (Proposition 4.1).

3. **Explicit Obstructions and IEEE-754 `float64` Cancellation Audit (Sections 5.3 & 6):**
   Parity reduction holds for *any* real-entire function (including Davenport–Heilbronn $\Xi _ {\mathrm{DH}}$); on an explicit off-critical zero perturbation $F _ {\mathrm{off}}(z)$, $L _ 1(F _ {\mathrm{off}}; t _ *) = -5.10 \times 10^{-11} < 0$. Moreover, Higham's inner-product bound $\lvert\mathrm{fl}(x^\top y) - x^\top y\rvert \le \gamma _ {N _ u}\lvert x\rvert^\top\lvert y\rvert$ shows that `float64` quadrature of $\widehat{W}(d,t) = \mathrm{e}^{\pi t/2}\widetilde{W}(d,t)$ amplifies roundoff error as $u _ {\mathrm{mach}}\mathrm{e}^{+\pi t/4}$, reaching 4.51% error at $t = 55$ and 1,375% error at $t = 60$.

---

## Summary of Methodological Benchmarks

| Benchmark Tier | Domain / Targets | Verified Quantitative Result | Reference |
| :--- | :--- | :--- | :--- |
| **Canonical Wigner vs. CR Moments** | $(d,t) = (0.20, \gamma_1)$ (`mp.dps = 50`) | $\lvert W _ {\mathrm{Wig}} - W _ {\mathrm{CR}}\rvert = 8.23 \times 10^{-55}$, $\lvert\widetilde{W} _ {\mathrm{Wig}} - \widetilde{W} _ {\mathrm{CR}}\rvert = 4.57 \times 10^{-54}$ | Section 5.1 |
| **Interval Moment Enclosures (`mpmath.iv`)** | Zeros $\gamma_1, \dots, \gamma _ {10}$ ($\Xi(\underline{\gamma} _ k)\Xi(\overline{\gamma} _ k) < 0{}$) | $c _ 1(\boldsymbol{\gamma} _ k).a, c _ 3(\boldsymbol{\gamma} _ k).a, c _ 5(\boldsymbol{\gamma} _ k).a > 0$; $c _ 3/c _ 1 \in [0.1354, 1.1022]$ | Table 2 |
| **Non-Harmonic-Safe 2D MVT Collar Boxes** | $80$ boxes around $\gamma_1, \tau_1, \gamma_2, \tau_2, \gamma_3$ ($\tau_1 \approx 15.5857, \tau_2 \approx 22.0980$) | $80 / 80$ boxes satisfy $\widetilde{W}(\mathbf{B} _ m).a \ge +1.5592 \times 10^{-12} > 0$ | Table 3 (Part B) |
| **Integer Argument Principle Winding** | $5$ rectangles $\Omega_1, \dots, \Omega_5$ on $[10^{-6}, 1.20] \times [0, 55]$ | $\lvert N _ {\mathrm{zeros}}(\Omega_k)\rvert \le 2.62 \times 10^{-14} < 0.5 \implies N _ {\mathrm{zeros}}(\Omega_k) = 0$ | Table 3 (Part C) |
| **IEEE-754 `float64` Cancellation Audit** | $d = 0.05$, $t \in [10, 60]$ ($N _ u = 4096$ vs. 50-dps `mpmath`) | Rel. error grows $\propto u _ {\mathrm{mach}}\mathrm{e}^{+\pi t/4}$: 4.51% at $t=55$, 1,375% at $t=60$ | Table 4 |

## Repository Structure

```text
.
├── paper/
│   ├── paper2_parity_reduced_wronskian.tex        # AMS-LaTeX manuscript source (13 pages)
│   ├── paper2_parity_reduced_wronskian.pdf        # Compiled pdfTeX publication PDF
│   └── figures/
│       ├── fig1_parity_reduced_wronskian_and_dipoles.pdf
│       ├── fig1_parity_reduced_wronskian_and_dipoles.png
│       ├── fig2_interval_mvt_and_argument_principle.pdf
│       └── fig2_interval_mvt_and_argument_principle.png
├── scripts/
│   └── verify_paper2_parity_reduced_wronskian.py  # Self-contained mpmath.iv + float64 audit suite
├── .agents/skills/parity-reduced-wronskian-xi/
│   └── SKILL.md                                   # AI Agent mathematical skill definition
├── AGENT_SKILL_MANIFEST.json                      # Machine-readable skill & theorem manifest
├── CITATION.cff                                   # Citation File Format metadata
├── .zenodo.json                                   # Zenodo archival metadata
└── LICENSE                                        # MIT License
```

## Reproducing the Numerical Verification & Figures

```bash
pip install numpy mpmath matplotlib torch
python scripts/verify_paper2_parity_reduced_wronskian.py
```

## License & Citation

Released under the [MIT License](LICENSE). Please cite this work using the metadata in [`CITATION.cff`](CITATION.cff).
