---
name: parity-reduced-wronskian-xi
description: >-
  Mathematical knowledge and verification skill for Paper 2 ("Parity-Reduced Transverse
  Wronskians, Higher Dipole Invariants, and Rigorous Continuum Certification for Riemann's
  Xi-Function"). Activate this skill when analyzing, extending, or numerically certifying the
  parity-reduced Wronskian W_tilde(d,t) = W(d,t)/d, the division-free hyperbolic-sinc kernel
  sinc_hyp(dy) = sinh(dy)/(dy), the higher transverse dipole invariants c_{2m-1}(t),
  directed-rounding mpmath.iv enclosures, 2D Interval-MVT boxes, the integer argument
  principle, or 2D continuum Lipschitz certification.
---

# Parity-Reduced Transverse Wronskian & Continuum Certification Skill (`parity-reduced-wronskian-xi`)

This skill provides AI research agents with the exact analytical identities, division-free Cauchy–Riemann gradient formulas, directed-rounding interval enclosures (`mpmath.iv`), and GPU-accelerated two-dimensional continuum certification procedures established in Paper 2.

---

## 1. Odd Transverse Parity and Division-Free Hyperbolic-Sinc Reduction

1. **Completed Riemann $\Xi$-Function and Transverse Wronskian:**
   With $s = \frac{1}{2} + d + \mathrm{i}t$ ($d = \sigma - \frac{1}{2}$) and $z = t - \mathrm{i}d$, $\Xi(t-\mathrm{i}d) = R(d,t) + \mathrm{i}I_{\mathrm{im}}(d,t)$ satisfies Schwarz reflection $\Xi(t+\mathrm{i}d) = \overline{\Xi(t-\mathrm{i}d)}$, so $\lvert\Xi(t-\mathrm{i}d)\rvert^2$ is even in $d$ and the transverse Wronskian

   $$W(d,t) := \frac{1}{2}\partial_d\lvert\Xi(t-\mathrm{i}d)\rvert^2 = R\,\partial_t I_{\mathrm{im}} - I_{\mathrm{im}}\,\partial_t R = -W(-d,t)$$

   is an odd real-analytic function of $d$ with $\partial_d^{2k}W(0,t) \equiv 0$ for all $k \ge 0$.

2. **Parity-Reduced Wronskian $\widetilde{W}(d,t)$ and Neumann Boundary Condition:**
   Factoring out $d$ via the entire, strictly positive, monotonically increasing hyperbolic-sinc kernel $\mathrm{sinc}_{\mathrm{hyp}}(w) := \frac{\sinh w}{w} = \sum_{m=0}^\infty \frac{w^{2m}}{(2m+1)!} \ge 1$ removes the $d \to 0^+$ boundary-layer degeneracy:

   $$\widetilde{W}(d,t) := \frac{W(d,t)}{d} = \frac{1}{2}\int_0^\infty y^2\,\mathrm{sinc}_{\mathrm{hyp}}(d y)\,I(y,t)\,\mathrm{d}y, \qquad \partial_d\widetilde{W}(0,t) \equiv 0,$$

   with critical-line boundary value given by the Lagarias–Turán invariant $\widetilde{W}(0,t) = (\Xi'(t))^2 - \Xi(t)\Xi''(t)$ and $\widetilde{W}(0,\gamma_k) = (\Xi'(\gamma_k))^2 > 0$ at every simple real zero $\gamma_k$.

## 2. Higher Transverse Dipole Hierarchy and Division-Free Cauchy–Riemann Gradients

1. **Transverse Dipole Moments $c_{2m-1}(t)$ and Pure-Square Coalescence Cascade:**
   The Taylor series $\widetilde{W}(d,t) = \sum_{m=1}^\infty c_{2m-1}(t)\,d^{2m-2}$ has coefficients

   $$c_{2m-1}(t) := \frac{1}{(2m-1)!}\partial_d^{2m-1}W(0,t) = \frac{1}{2(2m-1)!}\int_0^\infty y^{2m}I(y,t)\,\mathrm{d}y = m\sum_{j=0}^{2m}\frac{(-1)^{m+j}}{j!\,(2m-j)!}\Xi^{(j)}(t)\Xi^{(2m-j)}(t).$$

   In particular, $c_3(t) = \frac{1}{6}(\Xi\Xi^{(4)} - 4\Xi'\Xi''' + 3(\Xi'')^2)$ and $c_5(t) = \frac{1}{120}(-\Xi\Xi^{(6)} + 6\Xi'\Xi^{(5)} - 15\Xi''\Xi^{(4)} + 10(\Xi''')^2)$. At an $m$-fold coalesced zero $t_0$, $c_1(t_0) = \dots = c_{2m-3}(t_0) = 0$ and $c_{2m-1}(t_0) = \frac{m}{(m!)^2}(\Xi^{(m)}(t_0))^2 > 0$.

2. **Division-Free Cauchy–Riemann Partial Derivatives:**
   With $\eta_{\mathrm{hyp}}(w) := \frac{w\cosh w - \sinh w}{w^3} = \sum_{k=0}^\infty \frac{2k+2}{(2k+3)!}w^{2k} \ge \frac{1}{3}$ and $I_{\mathrm{im}}(d,t) = d\,\widetilde{I}(d,t)$,

   $$\widetilde{W} = R\widetilde{I}_t - \widetilde{I}R_t, \qquad \partial_t\widetilde{W} = R\widetilde{I}_{tt} - \widetilde{I}R_{tt}, \qquad \partial_d\widetilde{W} = d\left(\widetilde{I}_t^2 - \widetilde{I}\widetilde{I}_{tt} + R\widehat{I}_{td} - \widehat{I}_d R_t\right).$$

## 3. Verification Execution

Run the standalone directed-rounding interval (`mpmath.iv`) and GPU continuum verification suite from the repository root:

```bash
python scripts/verify_paper2_parity_reduced_wronskian.py
```
