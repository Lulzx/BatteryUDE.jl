# ============================================================================
# src/parameters.jl
# Physical Constants, Cell Parameters (Chen2020 NMC811 / Graphite 5 Ah),
# and Dimensionless State Layout Helpers
# ============================================================================

const F_CONST = 96485.33212       # Faraday constant [C / mol]
const R_GAS   = 8.314462618       # Universal gas constant [J / (mol * K)]
const T_REF   = 298.15            # Reference temperature [K]
const T_SCALE = 10.0              # Temperature scaling factor [K]
const DELTA_SEI_SCALE = 1.0e-7    # SEI thickness scaling factor [m]
const J_DEG_SCALE     = 1.0e-3    # Parasitic current scaling factor [A / m^2]

"""
    CellParams{T<:Real}

Physical, geometric, transport, kinetic, thermal, and discretization parameters
for a Single Particle Model with Electrolyte and Thermal coupling (SPMeT),
parameterized on element type `T` for ForwardDiff / Zygote compatibility.
Defaults correspond to an NMC811 / Graphite 5.0 Ah cell (adapted from Chen et al., 2020).
"""
Base.@kwdef struct CellParams{T<:Real}
    # Discretization grid sizes
    N_r::Int = 12                 # Radial shells per electrode
    N_n::Int = 6                  # Axial control volumes in anode
    N_s::Int = 4                  # Axial control volumes in separator
    N_p::Int = 6                  # Axial control volumes in cathode

    # Micro-scale spherical particle radii [m]
    R_n::T = 5.86e-6              # Anode particle radius [m]
    R_p::T = 5.22e-6              # Cathode particle radius [m]

    # Macro-scale axial domain thicknesses [m]
    L_n::T = 85.2e-6              # Anode thickness [m]
    L_s::T = 12.0e-6              # Separator thickness [m]
    L_p::T = 75.6e-6              # Cathode thickness [m]

    # Cell active cross-sectional area [m^2] & nominal 1C current [A]
    A_cell::T = 0.135             # Active electrode area [m^2] (~5.0 Ah nominal capacity)
    I_1C::T   = 5.0               # Nominal 1C discharge current [A]

    # Solid-phase diffusivities [m^2 / s] & Arrhenius activation energies [J / mol]
    D_sn::T     = 3.3e-14         # Anode reference solid diffusivity [m^2 / s]
    D_sp::T     = 4.0e-14         # Cathode reference solid diffusivity [m^2 / s]
    E_a_dsn::T  = 35000.0         # Anode diffusion activation energy [J / mol]
    E_a_dsp::T  = 25000.0         # Cathode diffusion activation energy [J / mol]

    # Electrolyte transport parameters
    D_e::T      = 2.6e-10         # Bulk electrolyte diffusivity [m^2 / s]
    c_e0::T     = 1000.0          # Initial / reference electrolyte concentration [mol / m^3]
    t_plus::T   = 0.38            # Cation transference number t_+^0 [-]
    kappa_e::T  = 1.15            # Electrolyte ionic conductivity [S / m]

    # Maximum solid concentrations [mol / m^3]
    c_sn_max::T = 33133.0         # Maximum lithium concentration in graphite anode [mol / m^3]
    c_sp_max::T = 51555.0         # Maximum lithium concentration in NMC811 cathode [mol / m^3]

    # Initial volume fractions & porosities [-]
    eps_e_n0::T = 0.30            # Initial anode electrolyte porosity [-]
    eps_e_s::T  = 0.40            # Separator electrolyte porosity [-]
    eps_e_p::T  = 0.30            # Cathode electrolyte porosity [-]
    eps_s_n0::T = 0.60            # Initial anode active material volume fraction [-]
    eps_s_p::T  = 0.65            # Cathode active material volume fraction [-]

    # Reaction rate constants [m^2.5 / (mol^0.5 * s)]
    k_n::T      = 2.0e-11         # Anode Butler-Volmer reaction rate constant
    k_p::T      = 3.5e-11         # Cathode Butler-Volmer reaction rate constant

    # SEI & Ohmic resistance parameters
    M_sei::T     = 0.162          # SEI molar mass [kg / mol] (lithium ethylene dicarbonate)
    rho_sei::T   = 1690.0         # SEI mass density [kg / m^3]
    kappa_sei::T = 5.0e-6         # SEI film ionic conductivity [S / m]
    R_film0::T   = 1.0e-3         # Initial baseline anode film resistance [Ohm * m^2]
    R_ohm::T     = 2.0e-3         # Lumped collector/tab ohmic resistance [Ohm]

    # Thermal parameters
    m_cell::T    = 0.070          # Cell mass [kg]
    C_p::T       = 850.0          # Cell specific heat capacity [J / (kg * K)]
    h_conv::T    = 15.0           # Convective cooling coefficient [W / (m^2 * K)]
    A_surf::T    = 0.0085         # External cell cooling surface area [m^2]
    T_ref::T     = 298.15         # Reference temperature [K]
    T_amb::T     = 298.15         # Ambient cooling temperature [K]
    dUp_dT::T    = -0.0001        # Cathode entropic coefficient [V / K]
    dUn_dT::T    = 0.0001         # Anode entropic coefficient [V / K]
end

"""
    chen2020_params(; kwargs...)

Construct a `CellParams{Float64}` instance with Chen2020 NMC811/Graphite defaults,
allowing optional keyword overrides.
"""
chen2020_params(; kwargs...) = CellParams{Float64}(; kwargs...)

"""
    num_electrolyte_cells(cell::CellParams)

Total number of axial electrolyte FVM cells `N_x = N_n + N_s + N_p`.
"""
@inline num_electrolyte_cells(cell::CellParams) = cell.N_n + cell.N_s + cell.N_p

"""
    state_dimension(cell::CellParams)

Total ODE state dimension `D = 2 * N_r + N_x + 5`.
"""
@inline state_dimension(cell::CellParams) = 2 * cell.N_r + num_electrolyte_cells(cell) + 5

"""
    solid_diffusivity(D_ref, E_a, T_kelvin, T_ref)

Arrhenius temperature-dependent solid diffusivity `D_s(T)`.
"""
@inline function solid_diffusivity(D_ref, E_a, T_kelvin, T_ref)
    return D_ref * exp(-(E_a / R_GAS) * (inv(T_kelvin) - inv(T_ref)))
end

"""
    initial_state(cell::CellParams{T};
                  theta_n0 = 0.85,
                  theta_p0 = 0.25,
                  c_e0_tilde = 1.0,
                  delta_sei0_tilde = 0.0,
                  eps_e_n0 = cell.eps_e_n0,
                  eps_s_n0 = cell.eps_s_n0,
                  Q_lli0 = 0.0,
                  T0_tilde = 0.0) where {T}

Construct the dimensionless initial state vector `u0` of length `D = 2N_r + N_x + 5`:
- `u[1 : N_r]`           : `c_tilde_sn` (dimensionless anode solid concentration)
- `u[N_r+1 : 2N_r]`      : `c_tilde_sp` (dimensionless cathode solid concentration)
- `u[2N_r+1 : 2N_r+N_x]` : `c_tilde_e`  (dimensionless electrolyte concentration)
- `u[2N_r+N_x+1]`        : `delta_sei_tilde` (dimensionless SEI film thickness)
- `u[2N_r+N_x+2]`        : `eps_e_n` (anode electrolyte porosity, default 0.30)
- `u[2N_r+N_x+3]`        : `eps_s_n` (anode active material volume fraction, default 0.60)
- `u[2N_r+N_x+4]`        : `Q_lli` (cumulative lost lithium inventory [A*h], default 0.0)
- `u[2N_r+N_x+5]`        : `T_tilde` (dimensionless cell temperature, default 0.0)
"""
function initial_state(
    cell::CellParams{T};
    theta_n0::Real = 0.85,
    theta_p0::Real = 0.25,
    c_e0_tilde::Real = 1.0,
    delta_sei0_tilde::Real = 0.0,
    eps_e_n0::Real = cell.eps_e_n0,
    eps_s_n0::Real = cell.eps_s_n0,
    Q_lli0::Real = 0.0,
    T0_tilde::Real = 0.0,
) where {T<:Real}
    N_r = cell.N_r
    N_x = num_electrolyte_cells(cell)
    D   = state_dimension(cell)

    u0 = zeros(T, D)
    u0[1:N_r]               .= T(theta_n0)
    u0[(N_r + 1):(2 * N_r)] .= T(theta_p0)
    u0[(2 * N_r + 1):(2 * N_r + N_x)] .= T(c_e0_tilde)
    u0[2 * N_r + N_x + 1]   = T(delta_sei0_tilde)
    u0[2 * N_r + N_x + 2]   = T(eps_e_n0)
    u0[2 * N_r + N_x + 3]   = T(eps_s_n0)
    u0[2 * N_r + N_x + 4]   = T(Q_lli0)
    u0[2 * N_r + N_x + 5]   = T(T0_tilde)
    return u0
end
