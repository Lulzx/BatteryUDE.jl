# BatteryUDE.jl — Differentiable SPMeT & Universal Differential Equations for Battery Degradation

[![SciML](https://img.shields.io/badge/SciML-Universal_DiffEqs-9558B2.svg)](https://sciml.ai/)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-389826.svg)](https://julialang.org/)
[![Tests](https://img.shields.io/badge/Tests-10%2F10_Passing-10b981.svg)](#verification--benchmark-metrics)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**`BatteryUDE.jl`** is a scientific machine learning (SciML) Julia package that embeds `Lux.jl` neural networks directly inside a Finite-Volume discretized **Single Particle Model with Electrolyte and Thermal coupling (SPMeT)** using **Universal Differential Equations (`OrdinaryDiffEq.jl` + `SciMLSensitivity.jl`)**.

It locks in exact physical conservation laws ($\left|\Delta N_{\text{Li}} / N_0\right| < 1.2 \times 10^{-15}$) while learning unknown, nonlinear Solid-Electrolyte Interphase (SEI) and Loss of Active Material (LAM) degradation kinetics directly from noisy voltage and capacity cycling measurements—and distilling the trained neural network back into closed-form analytical equations via sparse **SINDy (STLSQ)** regression.

<p align="center">
  <img src="docs/assets/architecture.svg" alt="BatteryUDE.jl Architecture" width="100%"/>
</p>

---

## Highlights

- **Exact Lithium Mass Conservation:** Conservative Spherical ($2 \times 12$ shells) and Harmonic-Mean Axial Electrolyte ($16$ cells) Finite Volume Method (FVM) stencils conserve solid and electrolyte lithium inventory to machine precision ($< 1.2 \times 10^{-15}$).
- **Thermodynamically Constrained Neural Surrogates:** Parasitic SEI reduction is parameterized via `Lux.Chain` and passed through `-j_scale * softplus(NN_θ(z_sei)) ≤ 0`, guaranteeing strictly cathodic side-reaction currents.
- **Exact Continuous Adjoint Gradients:** Mutation-free out-of-place `ude_rhs` and dual-number-safe promotion allow seamless differentiation with `InterpolatingAdjoint(autojacvec=ZygoteVJP())` and `Rodas5P()` (`cos_sim = 1.0`, relative error `4.0e-14` vs. `ForwardDiff`).
- **Symbolic SINDy Law Recovery:** Automatically extracts closed-form analytical Tafel-diffusion degradation equations from trained neural networks using column-normalized Sequential Thresholded Least Squares (`STLSQ`).

---

## Physics & Benchmark Results

### 1. Conservative SPMeT Forward Physics & 1C Discharge
<p align="center">
  <img src="docs/assets/spmet_physics.png" alt="SPMeT Forward Physics & Mass Conservation" width="100%"/>
</p>

### 2. Multi-Cycle UDE Digital Twin Training, Extrapolation & Symbolic SINDy Recovery
<p align="center">
  <img src="docs/assets/ude_training_sindy.png" alt="UDE Training, Holdout Extrapolation, and SINDy Distillation" width="100%"/>
</p>

---

## Verification & Benchmark Metrics

| Verification Gate | Target Criterion | Measured Result | Status |
| :--- | :--- | :--- | :---: |
| **Solid-Phase $\text{Li}^+$ Mass Conservation** (`test_conservation.jl`) | Relative error $< 10^{-12}$ | **`1.166 × 10⁻¹⁵`** | ✅ |
| **Electrolyte-Phase $\text{Li}^+$ Mass Conservation** (`test_conservation.jl`) | Relative error $< 10^{-12}$ | **`9.691 × 10⁻¹⁶`** | ✅ |
| **1C Constant-Current Discharge** (`test_forward_physics.jl`) | $V(0) \in [4.1, 4.3]\text{ V}$, monotonic to $2.8\text{ V}$ | **`4.101 V → 2.800 V`** ($t = 3637.7\text{ s}$) | ✅ |
| **Adjoint vs. ForwardDiff Gradients** (`test_adjoint_grads.jl`) | Cosine similarity $> 0.999$, Rel. error $< 10^{-4}$ | **`cos = 1.0`**, **`rel_err = 4.00 × 10⁻¹⁴`** | ✅ |
| **Two-Stage Curriculum Training** (`Adam` + `L-BFGS`) | Loss reduction $\ge 100\times$ ($2$ orders of mag.) | **`207.54×`** (`2.892 → 1.394 × 10⁻²`) | ✅ |
| **Unseen Holdout Cycles (Cycles 19–25)** | Holdout voltage $\text{RMSE} < 5.0\text{ mV}$ | **`0.759 mV`** (down from `25.51 mV`) | ✅ |
| **Symbolic SINDy Distillation (STLSQ)** | Recover hidden Tafel-diffusion SEI equation | **`j_deg = -0.0003064 exp(-αη_sei)/(1+δ̃_sei)`** ($R^2 = 0.962$) | ✅ |

---

## Quick Start

### Run the Test Suite
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Run End-to-End Synthetic Digital Twin Training & Symbolic Recovery
```bash
julia --project=. scripts/train_synthetic_twin.jl
```

### Programmatic Usage
```julia
using BatteryUDE, Random

rng  = Random.MersenneTwister(2026)
cell = chen2020_params() # NMC811 / Graphite 5.0 Ah cell (45 dimensionless ODE states)

# 1. Generate 25-cycle synthetic experimental dataset (1 mV Gaussian voltage noise)
dataset = generate_synthetic_twin_data(cell; N_cycles=25, train_cycles=18, rng=rng)

# 2. Initialize Coupled SPMeT + Lux.jl Neural SEI Model
model = HybridSPMeTModel(cell; rng=rng, hidden_dim=24, j_scale=0.30)

# 3. Train via Two-Stage Curriculum: Adam(5e-3, 100 iters) -> L-BFGS(40 iters)
train_res = train_battery_ude!(model, dataset; adam_lr=5e-3, adam_iters=100, lbfgs_iters=40)

# 4. Distill Trained Neural Surrogate into a Closed-Form Equation via SINDy (STLSQ)
sindy_res = distill_sei_sindy(model, train_res.p_trained)
println(sindy_res.equation_str)
# => j_deg(t) = -0.0003064 * exp(-alpha * eta_sei) / (1 + delta_sei_tilde)
```

---

## Mathematical Formulation

### State Vector Layout ($D = 2N_r + N_x + 5 = 45$)
All states are scaled to $\mathcal{O}(1)$ for stiff Rosenbrock (`Rodas5P`) conditioning:
- `u[1 : N_r]`: $\tilde{c}_{s,n} = c_{s,n} / c_{s,n}^{\max} \in (0, 1)$ — Dimensionless anode solid concentration ($N_r = 12$)
- `u[N_r+1 : 2N_r]`: $\tilde{c}_{s,p} = c_{s,p} / c_{s,p}^{\max} \in (0, 1)$ — Dimensionless cathode solid concentration ($N_r = 12$)
- `u[2N_r+1 : 2N_r+N_x]`: $\tilde{c}_e = c_e / c_{e,0} \approx 1.0$ — Dimensionless electrolyte concentration ($N_x = 6 + 4 + 6 = 16$)
- `u[2N_r+N_x+1]`: $\tilde{\delta}_{\text{sei}} = \delta_{\text{sei}} / (10^{-7}\text{ m})$ — Dimensionless SEI film thickness
- `u[2N_r+N_x+2]`: $\varepsilon_{e,n}$ — Anode electrolyte porosity (pore clogging)
- `u[2N_r+N_x+3]`: $\varepsilon_{s,n}$ — Anode active material volume fraction
- `u[2N_r+N_x+4]`: $Q_{\text{lli}}$ — Cumulative lost lithium inventory $[\text{A}\cdot\text{h}]$
- `u[2N_r+N_x+5]`: $\tilde{T} = (T - 298.15) / 10.0$ — Dimensionless cell temperature

### Coupled Neural SEI Kinetics
The neural network takes the normalized 4-element surface feature vector $\mathbf{z}_{\text{sei}} = \left[\theta_n^{\text{surf}}, \; j_{\text{tot},n}/10, \; (U_n(\theta_n^{\text{surf}}) + \eta_{\text{int},n}^{\text{approx}} - 0.4)/0.05, \; \tilde{\delta}_{\text{sei}}\right]^T$ and predicts the cathodic degradation current density:
$$j_{\text{deg}}(t) = -j_{\text{scale}} \cdot \text{softplus}\left(\mathcal{N}_{\theta}(\mathbf{z}_{\text{sei}})\right) \le 0, \qquad j_{\text{int},n} = j_{\text{tot},n} - j_{\text{deg}}$$
$$\frac{d\tilde{\delta}_{\text{sei}}}{dt} = -\frac{M_{\text{sei}}}{2 F \rho_{\text{sei}} \delta_{\text{scale}}} j_{\text{deg}}, \qquad \frac{d\varepsilon_{e,n}}{dt} = -a_{s,n} \delta_{\text{scale}} \frac{d\tilde{\delta}_{\text{sei}}}{dt}, \qquad \frac{dQ_{\text{lli}}}{dt} = -\frac{A_{\text{cell}} L_n a_{s,n}}{3600} j_{\text{deg}}$$

---

## Repository Structure

```text
BatteryUDE.jl/
├── Project.toml
├── README.md
├── spec.md
├── docs/
│   └── assets/
│       ├── architecture.svg
│       ├── spmet_physics.png
│       └── ude_training_sindy.png
├── src/
│   ├── BatteryUDE.jl              # Main module exports
│   ├── parameters.jl              # CellParams struct & Chen2020 (NMC811/Graphite) defaults
│   ├── fvm_discretization.jl      # Conservative Spherical & Axial FVM stencils
│   ├── electrochemistry.jl        # OCP curves, Butler-Volmer kinetics, Terminal Voltage
│   ├── neural_surrogates.jl       # Lux.jl SEI & LAM neural architectures
│   ├── hybrid_ode.jl              # Coupled SPMeT + UDE ODEProblem & Voltage Cutoff Callbacks
│   ├── training.jl                # Adjoint loss, Synthetic Twin generator, Adam + L-BFGS loop
│   └── sindy_distillation.jl      # Sparse symbolic polynomial/exponential distillation of NN
├── scripts/
│   └── train_synthetic_twin.jl    # End-to-end runnable script demonstrating law recovery
└── test/
    ├── runtests.jl
    ├── test_conservation.jl       # Verifies exact Li mass conservation (< 1e-12)
    ├── test_forward_physics.jl    # Verifies 1C discharge curve & voltage bounds
    └── test_adjoint_grads.jl      # Verifies Zygote/InterpolatingAdjoint vs ForwardDiff
```
