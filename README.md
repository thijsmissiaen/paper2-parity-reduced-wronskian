# Parity-Reduced Transverse Wronskians, Higher Dipole Invariants, and Rigorous Continuum Certification for Riemann's $\Xi$-Function

**Author:** Thijs Missiaen (Independent Researcher, Belgium · [ORCID: 0009-0007-3678-0810](https://orcid.org/0009-0007-3678-0810) · `thijs.missiaen@gmail.com`)  
**Series:** Transverse Wronskian of Riemann's $\Xi$-Function (Paper II of IX · [Paper I DOI: 10.5281/zenodo.23637324](https://doi.org/10.5281/zenodo.23637324))  
**Repository:** [`https://github.com/thijsmissiaen/paper2-parity-reduced-wronskian`](https://github.com/thijsmissiaen/paper2-parity-reduced-wronskian)  
**MSC 2020:** Primary `11M06`, `65G30`; Secondary `30D15`, `26D15`, `11Y35`  
**Manuscript:** [`paper/paper2_parity_reduced_wronskian.pdf`](paper/paper2_parity_reduced_wronskian.pdf) (15 pages, AMS-LaTeX / pdfTeX)

---

## Abstract & Overview

Let $\xi(s) = \frac{1}{2}s(s-1)\pi^{-s/2}\Gamma(s/2)\zeta(s)$ be Riemann's completed zeta function, and set $s = \frac{1}{2} + d + \mathrm{i}t$ with transverse offset $d = \sigma - \frac{1}{2} \in \mathbb{R}$ and spectral height $t \in \mathbb{R}$. Writing $z = t - \mathrm{i}d$ and $\Xi(z) := \xi(\frac{1}{2} + \mathrm{i}z) = R(d,t) + \mathrm{i}I _ {\mathrm{im}}(d,t)$, the equivalence theorems of Speiser (1934), Hinkkanen (1997), and Lagarias (1999) link zero-free regions in the half-plane $d > 0$ to the positivity of the transverse Wronskian

$$W(d,t) := \frac{1}{2}\frac{\partial}{\partial d}\lvert\Xi(t-\mathrm{i}d)\rvert^2 = R \partial_t I _ {\mathrm{im}} - I _ {\mathrm{im}} \partial_t R.$$

A classical obstacle in two-dimensional numerical and interval-arithmetic certification of the critical strip is the *boundary-layer degeneracy* as $d \to 0^+$: because $\Xi(t)$ is real-valued on the critical line $d = 0$, $I _ {\mathrm{im}}(0,t) \equiv 0$ and $\partial_d R(0,t) \equiv 0$, forcing $W(0,t) \equiv 0$ identically for all $t \in \mathbb{R}$.

In this paper we resolve this boundary-layer singularity analytically and construct a division-free directed-rounding interval and two-dimensional continuum certification architecture across the critical strip:

1. **Odd Transverse Parity and Division-Free Hyperbolic-Sinc Reduction (Section 2, Proposition 2.1):**
   Using the Schwarz reflection symmetry $\Xi(t+\mathrm{i}d) = \overline{\Xi(t-\mathrm{i}d)}$, we prove that $W(-d,t) = -W(d,t)$ is an odd real-analytic function of $d$ whose even transverse derivatives vanish identically on the critical line: $\partial_d^{2k}W(0,t) \equiv 0$ for all $k \ge 0$. Factoring $d$ analytically inside the Wigner phase-space integral via the strictly positive, monotonically increasing hyperbolic-sinc kernel $\mathrm{sinc} _ {\mathrm{hyp}}(w) := \frac{\sinh w}{w} = \sum _ {m=0}^\infty \frac{w^{2m}}{(2m+1)!} \ge 1$ yields the parity-reduced Wronskian

$$\widetilde{W}(d,t) := \frac{W(d,t)}{d} = \frac{1}{2}\int_0^\infty y^2 \mathrm{sinc} _ {\mathrm{hyp}}(d y) I(y,t) \mathrm{d}y, \qquad \partial_d\widetilde{W}(0,t) \equiv 0,$$

   whose critical-line boundary value is the Lagarias–Turán quadratic form $\widetilde{W}(0,t) = (\Xi'(t))^2 - \Xi(t)\Xi''(t)$, reducing at every simple real zero $\gamma_k$ to $\widetilde{W}(0,\gamma_k) = (\Xi'(\gamma_k))^2 > 0$.

2. **Higher Transverse Dipole Hierarchy and Pure-Square Coalescence Cascade (Section 3, Theorem 3.1 & Proposition 3.3):**
   Expanding $\widetilde{W}(d,t) = \sum _ {m=1}^\infty c _ {2m-1}(t) d^{2m-2}$ via Leibniz's rule and Cauchy–Riemann relations establishes closed-form differential and Wigner moment identities for all higher transverse dipole invariants:

$$c _ {2m-1}(t) := \frac{1}{(2m-1)!}\partial_d^{2m-1}W(0,t) = \frac{1}{2(2m-1)!}\int_0^\infty y^{2m}I(y,t) \mathrm{d}y = m\sum _ {j=0}^{2m}\frac{(-1)^{m+j}}{j! (2m-j)!}\Xi^{(j)}(t)\Xi^{(2m-j)}(t).$$

   If $t_0 \in \mathbb{R}$ is an $m$-fold coalesced zero ($\Xi(t_0) = \dots = \Xi^{(m-1)}(t_0) = 0$, $\Xi^{(m)}(t_0) \neq 0$), then $c_1(t_0) = \dots = c _ {2m-3}(t_0) = 0$ and the leading non-zero coefficient is the strictly positive pure square $c _ {2m-1}(t_0) = \frac{m}{(m!)^2}(\Xi^{(m)}(t_0))^2 > 0$. Moreover, introducing $\eta _ {\mathrm{hyp}}(w) := \frac{w\cosh w - \sinh w}{w^3} \ge \frac{1}{3}$ yields division-free integral formulas for $\widetilde{W}(d,t)$, $\partial_t\widetilde{W}(d,t)$, and $\partial_d\widetilde{W}(d,t)$ free of $0/0$ cancellation at $d = 0$.

3. **Directed-Rounding Interval Arithmetic, 2D Interval-MVT Collar Boxes, and Continuum Certification (Sections 4–5 & Appendix A):**
   Using two independent evaluators in directed-rounding interval arithmetic (`mpmath.iv`) — Euler–Maclaurin summation with exact rational Bernoulli numbers and division-free Jacobi-theta quadrature with analytical truncation bounds $\varepsilon _ {\mathrm{trunc}} \le 1.01 \times 10^{-55}$ — we certify $c_1(\gamma_k) > 0$, $c_3(\gamma_k) > 0$, and $c_5(\gamma_k) > 0$ at the first ten non-trivial zeros $\gamma_1, \dots, \gamma _ {10}$, enclose $\widetilde{W}(\mathbf{B} _ m).a > 0$ across $80$ two-dimensional Interval-MVT collar boxes, prove $N _ {\mathrm{zeros}}(\Omega_k) = 0$ on five macro-rectangles covering $[10^{-6}, 1.20] \times [0, 55]$ via the integer argument principle, and certify $\widehat{W} _ {\mathrm{lower}}(C _ {i,j}) > 0$ across all $4{,}987{,}101$ continuum Lipschitz cells covering $[0, 0.50] \times [0, 55]$.

---

## Summary of Certified Benchmarks

| Benchmark Tier | Domain / Targets | Certified Quantitative Result | Reference |
| :--- | :--- | :--- | :--- |
| **Wigner vs. Cauchy–Riemann Equivalence** | $(d,t) = (0.20, \gamma_1)$ (`mp.dps = 50`) | $\lvert\widetilde{W} _ {\mathrm{Wig}} - \widetilde{W} _ {\mathrm{CR}}\rvert = 4.57 \times 10^{-54}$, $\lvert c _ {3,\mathrm{Wig}} - c _ {3,\mathrm{CR}}\rvert = 1.19 \times 10^{-55}$ | Section 5.1 |
| **Interval Dipole Certification (`mpmath.iv`)** | Zeros $\gamma_1, \dots, \gamma _ {10}$ ($\Xi(\underline{\gamma} _ k)\Xi(\overline{\gamma} _ k) < 0$) | $c_1(\boldsymbol{\gamma} _ k).a > 0$, $c_3(\boldsymbol{\gamma} _ k).a > 0$, $c_5(\boldsymbol{\gamma} _ k).a > 0$; $c_3/c_1 \in [0.1354, 1.1022]$ | Table 2 |
| **Dual Evaluator Cross-Check (`mpmath.iv`)** | $7$ points across $[0, 0.50] \times [14.13, 49.77]$ | 100% interval overlap ($A \cap B \neq \emptyset$), widths $4.99 \times 10^{-38}$ to $4.04 \times 10^{-27}$ | Table 3 (Part A) |
| **2D Interval-MVT Collar Enclosures** | $80$ boxes across Zones 1–5 ($\gamma_1, \tau_1, \gamma_2, \tau_2, \gamma_3$) | $80 / 80$ boxes satisfy $\widetilde{W}(\mathbf{B} _ m).a \ge +1.5592 \times 10^{-12} > 0$ ($\widehat{W}.a \ge +8.29 \times 10^3$) | Table 3 (Part B) |
| **Integer Argument Principle Winding** | $5$ macro-rectangles $\Omega_1, \dots, \Omega_5$ on $[10^{-6}, 1.20] \times [0, 55]$ | $\lvert N _ {\mathrm{zeros}}(\Omega_k)\rvert \le 2.62 \times 10^{-14} < 0.5 \implies N _ {\mathrm{zeros}}(\Omega_k) = 0$ | Table 3 (Part C) |
| **2D Continuum Lipschitz Certification** | $4{,}987{,}101$ cells ($400 \times 12{,}500$ grid on $[0, 0.50] \times [0, 55]$) | $4{,}987{,}101 / 4{,}987{,}101$ cells (100%) satisfy $\widehat{W} _ {\mathrm{lower}}(C _ {i,j}) > 0$; max margin 0.37% | Table 4 |

## Repository Structure

```text
.
├── paper/
│   ├── paper2_parity_reduced_wronskian.tex        # AMS-LaTeX manuscript source (15 pages)
│   ├── paper2_parity_reduced_wronskian.pdf        # Compiled pdfTeX publication PDF
│   └── figures/
│       ├── fig1_parity_reduced_wronskian_and_dipoles.pdf
│       ├── fig1_parity_reduced_wronskian_and_dipoles.png
│       ├── fig2_interval_mvt_and_argument_principle.pdf
│       └── fig2_interval_mvt_and_argument_principle.png
├── scripts/
│   └── verify_paper2_parity_reduced_wronskian.py  # Self-contained mpmath.iv + GPU verification suite
├── .agents/skills/parity-reduced-wronskian-xi/
│   └── SKILL.md                                   # AI Agent mathematical skill definition
├── AGENT_SKILL_MANIFEST.json                      # Machine-readable skill & theorem manifest
├── CITATION.cff                                   # Citation File Format metadata
├── .zenodo.json                                   # Zenodo archival metadata
└── LICENSE                                        # MIT License
```

## Reproducing the Numerical Verification & Figures

To execute the complete 4-tier verification suite (50-decimal Wigner moment equivalence, `mpmath.iv` directed-rounding dipole certification at $\gamma_1, \dots, \gamma _ {10}$, dual Euler–Maclaurin vs. Jacobi-theta evaluator cross-check, 80 2D Interval-MVT collar boxes, 5 macro argument-principle rectangles, 4,987,101-cell GPU continuum Lipschitz audit, and regeneration of Figures 1 and 2):

```bash
pip install numpy mpmath matplotlib torch
python scripts/verify_paper2_parity_reduced_wronskian.py
```

## License & Citation

Released under the [MIT License](LICENSE). Please cite this work using the metadata in [`CITATION.cff`](CITATION.cff).
