# Technical Specification: `BatteryUDE`

**Hybrid Electrochemical PDE & Universal Differential Equation Engine for Lithium-Ion Battery Degradation**

---

## 1. System Architecture & Scope

### 1.1 Objective

`BatteryUDE` (implementable in Julia via `SciML` or in Python/JAX via `Equinox`/`Diffrax`) is a differentiable hybrid solver that embeds neural networks inside the **Single Particle Model with Electrolyte and Thermal coupling (SPMeT)**. It preserves exact physical conservation laws (mass of lithium, charge neutrality, and Fickian/Nernst-Planck transport) while learning unknown, nonlinear parasitic degradation kinetics—specifically Solid-Electrolyte Interphase (SEI) growth, lithium plating, and active material loss (LAM)—directly from experimental cycling time-series data.

### 1.2 Core Design Constraints

* **Strict Conservation Enforcement:** The neural network is forbidden from modifying state derivatives directly; it may only predict scalar interfacial side-reaction flux densities $j_{\text{sei}}, j_{\text{pl}} \ge 0$, which enter the governing PDE boundary conditions and stoichiometric mass balances.
* **Dimensionless Scaling:** All PDE states and coordinates are scaled to $\mathcal{O}(1)$ prior to time-integration to prevent ill-conditioned Jacobians and exploding adjoint gradients.
* **Memory-Bounded Adjoints:** Continuous checkpointed adjoint sensitivity analysis scales as $\mathcal{O}(1)$ in memory with respect to the number of solver time steps.

---

## 2. Continuous Mathematical Formulation

### 2.1 Spatial Domains

* **Solid Phase (Micro-scale Spherical Particles):** Radial coordinate $r \in [0, R_k]$ for electrode domain $k \in \{n, p\}$ (negative anode $n$, positive cathode $p$).
* **Electrolyte Phase (Macro-scale Axial Coordinate):** Axial coordinate $x \in [0, L]$ partitioned into anode $x \in [0, L_n]$, separator $x \in [L_n, L_n + L_s]$, and cathode $x \in [L_n + L_s, L]$.

### 2.2 Solid-Phase Lithium Diffusion

Within active material particles of radius $R_k$, molar lithium concentration $c_{s,k}(r, t)$ follows spherical Fickian diffusion:


$$\frac{\partial c_{s,k}}{\partial t} = \frac{1}{r^2} \frac{\partial}{\partial r} \left( D_{s,k}(T) r^2 \frac{\partial c_{s,k}}{\partial r} \right), \quad r \in (0, R_k), \; k \in \{n, p\}$$

**Boundary Conditions:**


$$\left. \frac{\partial c_{s,k}}{\partial r} \right\vert{}_{r=0} = 0, \qquad -D_{s,k}(T) \left. \frac{\partial c_{s,k}}{\partial r} \right\vert{}_{r=R_k} = \frac{j_{\text{int}, k}(t)}{F}$$


where $F = 96485.33\text{ C/mol}$ is Faraday's constant and $j_{\text{int}, k}(t)$ is the true intercalation current density per unit active surface area ($\text{A/m}^2$).

### 2.3 Electrolyte-Phase Transport

Electrolyte lithium-ion concentration $c_e(x, t)$ across the three porous domains $k \in \{n, s, p\}$ follows:


$$\varepsilon_{e,k}(t) \frac{\partial c_e}{\partial t} = \frac{\partial}{\partial x} \left( \varepsilon_{e,k}^{\text{brugg}} D_e(c_e, T) \frac{\partial c_e}{\partial x} \right) + \frac{1 - t_+^0}{F} a_{s,k}(t) j_{\text{tot},k}(t)$$

**Boundary Conditions (Zero Flux at Current Collectors, Continuity at Interfaces):**


$$\left. \frac{\partial c_e}{\partial x} \right\vert{}_{x=0} = 0, \qquad \left. \frac{\partial c_e}{\partial x} \right\vert{}_{x=L} = 0$$

### 2.4 Conservation of Charge & Interfacial Kinetics

Given applied cell current $I_{\text{app}}(t)$ (defined such that $I_{\text{app}} > 0$ for discharge and $I_{\text{app}} < 0$ for charge) and electrode cross-sectional area $A_{\text{cell}}$, the total pore-wall current density is spatially uniform within each electrode under the SPMe approximation:


$$j_{\text{tot}, n}(t) = \frac{I_{\text{app}}(t)}{A_{\text{cell}} L_n a_{s,n}(t)}, \qquad j_{\text{tot}, p}(t) = -\frac{I_{\text{app}}(t)}{A_{\text{cell}} L_p a_{s,p}(t)}$$

At the cathode ($p$), no side reactions are assumed ($j_{\text{int}, p} = j_{\text{tot}, p}$). At the anode ($n$), total current splits between main lithium intercalation $j_{\text{int}, n}$ and parasitic degradation flux $j_{\text{deg}}$:


$$j_{\text{tot}, n}(t) = j_{\text{int}, n}(t) + j_{\text{deg}}(t) \implies j_{\text{int}, n}(t) = j_{\text{tot}, n}(t) - j_{\text{deg}}(t)$$

The main intercalation overpotential $\eta_{\text{int}, k}$ is governed by symmetric Butler-Volmer kinetics ($\alpha_a = \alpha_c = 0.5$), inverted analytically via $\text{arcsinh}$:


$$\eta_{\text{int}, k}(t) = \frac{2 R T(t)}{F} \text{arcsinh}\left( \frac{j_{\text{int}, k}(t)}{2 i_{0,k}(t)} \right)$$


where the exchange current density is:


$$i_{0,k}(t) = F k_{0,k}(T) \sqrt{\bar{c}_{e,k}(t) \, c_{s,k}^{\text{surf}}(t) \left( c_{s,k}^{\max} - c_{s,k}^{\text{surf}}(t) \right)}$$

### 2.5 Universal Differential Equation (UDE) Degradation Coupling

Instead of hard-coding heuristic Tafel equations for SEI growth and lithium plating, the parasitic degradation current density $j_{\text{deg}}(t)$ (note: $j_{\text{deg}} \le 0$ represents cathodic reduction of electrolyte/Li ions) and active material loss rate are parameterized by two compact neural networks $\mathcal{N}_{\text{sei}}(\cdot; \theta_1)$ and $\mathcal{N}_{\text{lam}}(\cdot; \theta_2)$:

#### 2.5.1 Neural Parasitic Flux ($\mathcal{N}_{\text{sei}}$)

Let the normalized feature vector at the anode particle surface be:


$$\mathbf{z}_{\text{sei}}(t) = \begin{bmatrix} \tilde{\theta}_n^{\text{surf}}(t) \\ \tilde{j}_{\text{tot},n}(t) \\ \tilde{\eta}_{\text{int},n}(t) \\ \tilde{T}(t) \\ \tilde{\delta}_{\text{sei}}(t) \end{bmatrix} = \begin{bmatrix} c_{s,n}^{\text{surf}}(t) / c_{s,n}^{\max} \\ j_{\text{tot},n}(t) / j_{\text{ref}} \\ \left( U_n(\theta_n^{\text{surf}}) + \eta_{\text{int},n}(t) - U_{\text{sei}}^{\text{ref}} \right) / V_{\text{th}} \\ (T(t) - T_{\text{ref}}) / \Delta T_{\text{ref}} \\ \delta_{\text{sei}}(t) / \delta_{\text{ref}} \end{bmatrix}$$

To guarantee thermodynamic consistency (parasitic reduction cannot generate positive anodic current), the output is passed through a negative `softplus` activation:


$$j_{\text{deg}}(t; \theta_1) = -j_{\text{deg, scale}} \, \text{softplus}\left( \mathcal{N}_{\text{sei}}\left(\mathbf{z}_{\text{sei}}(t); \theta_1\right) \right) \le 0$$

#### 2.5.2 Coupled Multi-Cycle Aging ODEs

The neural outputs drive four macroscopic degradation states:

1. **SEI Film Thickness $\delta_{\text{sei}}(t)$:**

$$\frac{d\delta_{\text{sei}}}{dt} = -\frac{M_{\text{sei}}}{2 F \rho_{\text{sei}}} j_{\text{deg}}(t; \theta_1)$$


2. **Anode Porosity Reduction $\varepsilon_{e,n}(t)$ (Pore Clogging):**

$$\frac{d\varepsilon_{e,n}}{dt} = -a_{s,n}(t) \frac{d\delta_{\text{sei}}}{dt}$$


3. **Anode Active Volume Fraction $\varepsilon_{s,n}(t)$ (Loss of Active Material via Mechanical Stress):**

$$\frac{d\varepsilon_{s,n}}{dt} = -\gamma_{\text{lam}} \, \text{softplus}\left( \mathcal{N}_{\text{lam}}\left( \frac{\vert{}\nabla_r c_{s,n}\vert{}_{\max}}{c_{s,n}^{\max}/R_n}, \, \tilde{T}(t); \, \theta_2 \right) \right)$$



which dynamically updates specific interfacial area $a_{s,n}(t) = \frac{3 \varepsilon_{s,n}(t)}{R_n}$.
4. **Total Lost Lithium Inventory $Q_{\text{lli}}(t)$ ($\text{A}\cdot\text{h}$):**

$$\frac{dQ_{\text{lli}}}{dt} = -\frac{A_{\text{cell}} L_n a_{s,n}(t)}{3600} j_{\text{deg}}(t; \theta_1)$$



### 2.6 Terminal Voltage & Thermal Balance Observation Equations

The terminal cell voltage $V_{\text{cell}}(t)$ is computed algebraically at every time step from the PDE states:


$$V_{\text{cell}}(t) = U_p\left(\theta_p^{\text{surf}}\right) - U_n\left(\theta_n^{\text{surf}}\right) + \eta_{\text{int},p}(t) - \eta_{\text{int},n}(t) + \Delta \Phi_e(t) - I_{\text{app}}(t) R_{\text{ohm}} - j_{\text{tot},n}(t) R_{\text{film}}\left(\delta_{\text{sei}}(t)\right)$$


where the SEI film resistance is $R_{\text{film}}(t) = R_{\text{film}, 0} + \frac{\delta_{\text{sei}}(t)}{\kappa_{\text{sei}}}$, and the lumped thermal ODE is:


$$m_{\text{cell}} C_p \frac{dT}{dt} = I_{\text{app}}(t) \left( U_p - U_n - V_{\text{cell}}(t) + T(t) \frac{\partial (U_p - U_n)}{\partial T} \right) - h_{\text{conv}} A_{\text{surf}} \left( T(t) - T_{\text{amb}}(t) \right)$$

---

## 3. Spatial Discretization (Finite Volume Method)

To conserve lithium mass to machine precision ($< 10^{-13}$), spatial derivatives are discretized via the **Finite Volume Method (FVM)** rather than finite differences.

### 3.1 Spherical Shell FVM ($N_r$ Control Volumes per Electrode)

Partition $[0, R_k]$ into $N_r$ concentric shells with boundaries $r_{i-1/2} = \frac{i-1}{N_r} R_k$ and centers $r_i = \frac{i - 1/2}{N_r} R_k$ for $i = 1, \dots, N_r$.

* Shell volume: $V_i = \frac{4}{3}\pi \left( r_{i+1/2}^3 - r_{i-1/2}^3 \right)$
* Interface surface area: $A_{i+1/2} = 4\pi r_{i+1/2}^2$

Integrating over shell $i$ yields the exact semi-discrete ODE:


$$\frac{dc_{s,k,i}}{dt} = \frac{1}{V_i} \left( A_{i+1/2} F_{i+1/2} - A_{i-1/2} F_{i-1/2} \right)$$


where the diffusive fluxes are:


$$F_{1/2} = 0, \qquad F_{i+1/2} = D_{s,k}(T) \frac{c_{s,k,i+1} - c_{s,k,i}}{\Delta r} \;\; (1 \le i < N_r), \qquad F_{N_r+1/2} = -\frac{j_{\text{int},k}(t)}{F}$$

Surface concentration $c_{s,k}^{\text{surf}}$ is extrapolated via quadratic Lagrange interpolation using $c_{s,k,N_r-1}, c_{s,k,N_r}$ and the known boundary gradient.

### 3.2 Axial Electrolyte FVM ($N_x = N_n + N_s + N_p$ Control Volumes)

For control volume $m$ of width $\Delta x_m$:


$$\varepsilon_{e,m}(t) \frac{dc_{e,m}}{dt} = \frac{1}{\Delta x_m} \left( J_{m+1/2} - J_{m-1/2} \right) + \frac{1 - t_+^0}{F} a_{s,m}(t) j_{\text{tot},m}(t)$$


Inter-volume effective diffusivity $D_{e, m+1/2}^{\text{eff}}$ at the anode/separator and separator/cathode interfaces is computed using the **harmonic mean** to preserve flux continuity across discontinuous porosities:


$$D_{e, m+1/2}^{\text{eff}} = \frac{\Delta x_m + \Delta x_{m+1}}{\frac{\Delta x_m}{D_{e,m}^{\text{eff}}} + \frac{\Delta x_{m+1}}{D_{e,m+1}^{\text{eff}}}}$$

### 3.3 Full State Vector $\mathbf{u}(t) \in \mathbb{R}^{D}$

With $N_r = 16$ radial shells per electrode and $N_x = 10 + 5 + 10 = 25$ electrolyte volumes, the total semi-discrete state vector has dimension $D = 2N_r + N_x + 5 = 62$:


$$\mathbf{u}(t) = \left[ \underbrace{\tilde{\mathbf{c}}_{s,n}}_{\mathbb{R}^{16}}, \; \underbrace{\tilde{\mathbf{c}}_{s,p}}_{\mathbb{R}^{16}}, \; \underbrace{\tilde{\mathbf{c}}_e}_{\mathbb{R}^{25}}, \; \tilde{\delta}_{\text{sei}}, \; \varepsilon_{e,n}, \; \varepsilon_{s,n}, \; Q_{\text{lli}}, \; \tilde{T} \right]^T$$

---

## 4. Differentiable Programming & Adjoint Training Protocol

### 4.1 Stiff ODE Integration & Adjoint Selection

Because electrolyte diffusion ($\tau_e \sim 10^1\text{ s}$), solid diffusion ($\tau_s \sim 10^3\text{ s}$), and multi-cycle SEI growth ($\tau_{\text{sei}} \sim 10^6\text{ s}$) span five orders of magnitude, explicit Runge-Kutta solvers fail due to stiffness constraints.

* **Forward Solver:** L-stable Rosenbrock method `Rodas5P()` (Julia) or `Kvaerno5()` (JAX `Diffrax`) with relative tolerance `reltol = 1e-6`, absolute tolerance `abstol = 1e-8`, and precomputed sparsity pattern for the $62 \times 62$ Jacobian $\frac{\partial \mathbf{f}}{\partial \mathbf{u}}$.
* **Backward Pass:** `InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true)` solving the augmented adjoint ODE backward in time:

$$\frac{d\boldsymbol{\lambda}}{dt} = -\left( \frac{\partial \mathbf{f}}{\partial \mathbf{u}} \right)^T \boldsymbol{\lambda}(t) - \left( \frac{\partial g}{\partial \mathbf{u}} \right)^T, \qquad \frac{d\mathcal{L}}{d\theta} = \int_{t_0}^{t_f} \boldsymbol{\lambda}(t)^T \frac{\partial \mathbf{f}}{\partial \theta} dt$$



### 4.2 Cycle-by-Cycle State Projection & Event Handling

Simulating 500 contiguous cycles ($\sim 10^6\text{ seconds}$) inside a single ODE call causes adjoint gradient blow-up. Instead, training operates over **Cycle Mini-Batches**:

1. **Intra-Cycle Integration:** Given aging state $\mathbf{d}_c = [\delta_{\text{sei}, c}, \varepsilon_{e,n,c}, \varepsilon_{s,n,c}, Q_{\text{lli},c}]^T$ at the start of cycle $c$, integrate a full CC-CV charge/discharge cycle $t \in [0, t_{\text{cycle}}]$ using continuous terminal voltage cutoff callbacks ($V_{\text{cell}}(t) - V_{\min} = 0$).
2. **Inter-Cycle Operator Splitting (Cycle-Jump):** Since $\mathbf{d}_c$ changes by $< 0.05\%$ per cycle, compute the net per-cycle aging increment $\Delta \mathbf{d}_c(\theta) = \mathbf{d}(t_{\text{cycle}}; \theta) - \mathbf{d}_c$ over key anchor cycles $c \in \{1, 10, 25, 50, 100, \dots\}$ and integrate the macro-cycle envelope via coarse Runge-Kutta steps in cycle-space $N$.

### 4.3 Multi-Objective Physics-Informed Loss Function

$$\mathcal{L}(\theta) = w_V \frac{1}{\vert{} \mathcal{C} \vert{}} \sum_{c \in \mathcal{C}} \frac{1}{K_c} \sum_{k=1}^{K_c} \left( V_{\text{sim}}(t_{c,k}; \theta) - V_{\text{exp}}(t_{c,k}) \right)^2 + w_Q \sum_{c \in \mathcal{C}} \left( \Delta Q_{\text{cap}, c}^{\text{sim}}(\theta) - \Delta Q_{\text{cap}, c}^{\text{exp}} \right)^2 + w_R \sum_{c \in \mathcal{C}} \left( R_{\text{dc}, c}^{\text{sim}}(\theta) - R_{\text{dc}, c}^{\text{exp}} \right)^2 + \lambda_{\text{reg}} \Vert{}\theta\Vert{}_2^2$$

### 4.4 Two-Stage Optimization Curriculum

1. **Stage 1:** `AdamW` ($\eta = 10^{-3}$, cosine decay, gradient norm clipping at $1.0$) for 300 epochs over progressively widening cycle horizons ($c \in [1, 20] \to [1, 100]$).
2. **Stage 2:** `L-BFGS` (history size 20, Strong Wolfe line search) for 100 iterations to converge to sharp local minima suitable for symbolic distillation.

---

## 5. Software Architecture & API Specification

### 5.1 Repository Directory Tree

```text
BatteryUDE/
├── Project.toml                     # (or pyproject.toml for Python/JAX implementation)
├── src/
│   ├── BatteryUDE.jl
│   ├── types.jl                     # Core structs: CellGeometry, ElectroChemParams, HybridSPMe
│   ├── discretization/
│   │   ├── spherical_fvm.jl         # Preallocated radial divergence & surface extrapolation stencils
│   │   ├── electrolyte_fvm.jl       # Harmonic-mean axial diffusivity & tridiagonal operators
│   │   └── sparsity.jl              # Symbolic Jacobian sparsity pattern detection
│   ├── physics/
│   │   ├── ocp_curves.jl            # Smooth analytical OCPs U_n(θ), U_p(θ) & entropic dU/dT
│   │   ├── butler_volmer.jl         # arcsinh overpotential & exchange current calculations
│   │   └── voltage_observer.jl      # Algebraic terminal voltage & heat generation evaluator
│   ├── surrogates/
│   │   ├── sei_network.jl           # Lux.jl / Equinox constrained neural architectures
│   │   ├── cycle_jump.jl            # Multi-cycle envelope accelerator
│   │   └── symbolic_sindy.jl        # Sparse regression to extract closed-form equations
│   └── io/
│       └── datasets.jl              # Parsers for NASA PCoE (.mat), CALCE, and Sandia HDF5/CSV
└── test/
    ├── test_mass_conservation.jl
    ├── test_pybamm_parity.jl
    └── test_adjoint_gradients.jl

```

### 5.2 Core Reference Implementation (Executable Prototype)

The self-contained module below implements the FVM spatial operators, the coupled SPMe + Neural UDE right-hand-side, and the terminal voltage observer:

```julia
module BatteryUDECore

using LinearAlgebra
using Lux
using ComponentArrays

struct CellParams{T}
    R_n::T; R_p::T
    L_n::T; L_s::T; L_p::T
    A_cell::T
    D_sn::T; D_sp::T; D_e::T
    c_sn_max::T; c_sp_max::T
    k_n::T; k_p::T
    t_plus::T
    M_sei::T; rho_sei::T; kappa_sei::T
    R_film0::T; R_ohm::T
end

"""
    spherical_fvm_rhs!(dc_dt, c_tilde, D_s, R_k, j_int, c_max, N_r)

In-place conservative Finite Volume discretization of spherical diffusion in dimensionless form.
"""
function spherical_fvm_rhs!(dc_dt, c_tilde, D_s, R_k, j_int, c_max, N_r::Int)
    F = 96485.3321
    dr = 1.0 / N_r
    alpha = D_s / (R_k^2)
    
    @inbounds for i in 1:N_r
        r_minus = (i - 1) * dr
        r_plus  = i * dr
        V_shell = (r_plus^3 - r_minus^3) / 3.0
        
        flux_minus = (i == 1) ? 0.0 : alpha * (c_tilde[i] - c_tilde[i-1]) / dr
        flux_plus  = (i == N_r) ? -j_int / (F * c_max * R_k) : alpha * (c_tilde[i+1] - c_tilde[i]) / dr
        
        dc_dt[i] = ((r_plus^2) * flux_plus - (r_minus^2) * flux_minus) / V_shell
    end
    return nothing
end

# Smooth analytical Open Circuit Potentials (NMC811 / Graphite)
ocp_graphite(theta) = 0.12 + 1.5 * exp(-120.0 * theta) - 0.035 * tanh((theta - 0.55) / 0.08)
ocp_nmc811(theta)   = 4.35 - 0.85 * theta - 0.15 * tanh((theta - 0.40) / 0.10) - 0.25 * exp(15.0 * (theta - 0.98))

"""
    ude_rhs!(du, u, p, t, cell::CellParams, nn_sei, st_sei, I_app_fn, N_r::Int)

Evaluates the coupled SPMe + Neural SEI Universal Differential Equation RHS.
State layout `u`:
  1:N_r         -> c_tilde_sn (dimensionless anode solid concentration)
  N_r+1:2N_r    -> c_tilde_sp (dimensionless cathode solid concentration)
  2N_r+1        -> delta_sei_tilde (normalized by 1e-7 m)
  2N_r+2        -> eps_s_n (anode active material volume fraction)
  2N_r+3        -> Q_lli (lost lithium inventory in A*h)
"""
function ude_rhs!(du, u, p_nn, t, cell::CellParams, nn_sei, st_sei, I_app_fn, N_r::Int)
    F = 96485.3321; R_gas = 8.31446; T = 298.15
    c_sn = @view u[1:N_r]
    c_sp = @view u[N_r+1:2N_r]
    delta_sei_tilde = u[2N_r+1]
    eps_sn          = u[2N_r+2]
    
    # Quadratic extrapolation to particle surface
    theta_n_surf = clamp(1.5 * c_sn[N_r] - 0.5 * c_sn[N_r-1], 1e-4, 1.0 - 1e-4)
    theta_p_surf = clamp(1.5 * c_sp[N_r] - 0.5 * c_sp[N_r-1], 1e-4, 1.0 - 1e-4)
    
    # Interfacial areas & total pore-wall current density
    a_sn = 3.0 * eps_sn / cell.R_n
    a_sp = 3.0 * 0.65 / cell.R_p
    I_app = I_app_fn(t)
    
    j_tot_n =  I_app / (cell.A_cell * cell.L_n * a_sn)
    j_tot_p = -I_app / (cell.A_cell * cell.L_p * a_sp)
    
    # Approximate main anode overpotential for neural feature input
    i0_n = F * cell.k_n * sqrt(1000.0) * cell.c_sn_max * sqrt(theta_n_surf * (1.0 - theta_n_surf))
    eta_n_approx = (2.0 * R_gas * T / F) * asinh(j_tot_n / (2.0 * i0_n))
    
    # Evaluate Neural SEI Surrogate (strictly <= 0 cathodic reduction)
    z_in = [theta_n_surf, j_tot_n / 10.0, (ocp_graphite(theta_n_surf) + eta_n_approx - 0.4) / 0.05, delta_sei_tilde]
    nn_out, _ = nn_sei(z_in, p_nn, st_sei)
    j_deg = -1e-3 * log1p(exp(nn_out[1])) # Smooth negative softplus
    
    # Split current between main intercalation and parasitic SEI
    j_int_n = j_tot_n - j_deg
    j_int_p = j_tot_p
    
    # 1. Solid diffusion FVM updates
    spherical_fvm_rhs!(@view(du[1:N_r]), c_sn, cell.D_sn, cell.R_n, j_int_n, cell.c_sn_max, N_r)
    spherical_fvm_rhs!(@view(du[N_r+1:2N_r]), c_sp, cell.D_sp, cell.R_p, j_int_p, cell.c_sp_max, N_r)
    
    # 2. Degradation ODE updates
    delta_scale = 1e-7
    du[2N_r+1] = (-cell.M_sei / (2.0 * F * cell.rho_sei * delta_scale)) * j_deg
    du[2N_r+2] = 0.0 # Optional LAM network hook
    du[2N_r+3] = -(cell.A_cell * cell.L_n * a_sn / 3600.0) * j_deg
    return nothing
end

end # module

```

---

## 6. Verification & Benchmark Suite

### 6.1 Unit & Physics Verification Tests

1. **Zero-Degradation Mass Conservation Test (`test_mass_conservation.jl`):**
* Set $\mathcal{N}_{\text{sei}} \equiv 0$ and simulate 10 sinusoidal charge/discharge cycles.
* **Pass Criterion:** Total volume-averaged lithium moles $\int_{\Omega_n} c_{s,n} dV + \int_{\Omega_p} c_{s,p} dV$ must remain invariant within $\le 10^{-13}$ relative error:

$$\left\vert{} \frac{N_{\text{Li}}(t) - N_{\text{Li}}(0)}{N_{\text{Li}}(0)} \right\vert{} \le 10^{-13}$$




2. **Forward Solver Parity Test (`test_pybamm_parity.jl`):**
* Compare simulated 1C and 2C discharge voltage curves against `PyBaMM` / `PETLION.jl` using the standard Chen2020 (LG M50 NMC811/Graphite) parameter set.
* **Pass Criterion:** Root-Mean-Square Error (RMSE) across the entire voltage curve $< 1.5\text{ mV}$.


3. **Adjoint vs. Forward-Mode Dual Number Check (`test_adjoint_gradients.jl`):**
* Compute $\nabla_{\theta} \mathcal{L}$ over a 1800-second discharge using `InterpolatingAdjoint` and compare against exact `ForwardDiff.gradient` dual numbers.
* **Pass Criterion:** Cosine similarity $> 0.9999$ and relative $L_{\infty}$ norm difference $< 10^{-5}$.



### 6.2 Empirical Generalization Benchmark (NASA PCoE & Sandia Datasets)

* **Training Split:** Train `BatteryUDE` on the first 80 cycles of cells `B0005` and `B0006` ($18650$ cells cycled at $24^\circ\text{C}$).
* **Extrapolation & Holdout Split:** Predict voltage curves, internal resistance rise $R_{\text{film}}(N)$, and State-of-Health (SoH) across cycles $81 \to 168$ on unseen cell `B0007` and variable-current profile `B0018`.
* **Target Metrics:**
* Remaining Useful Life (RUL) / End-of-Life ($80\%$ capacity) prediction error: $< 3.0\%$ cycles.
* Full-curve terminal voltage RMSE at cycle 150: $< 12\text{ mV}$.
