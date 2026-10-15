# ============================================================================
# src/hybrid_ode.jl
# Coupled SPMeT + Neural UDE Right-Hand-Side, HybridSPMeTModel,
# ODEProblem Builders, and Continuous Voltage Cutoff Callbacks
# ============================================================================

"""
    HybridSPMeTModel{C<:CellParams, N, S, A}

Container coupling physical `CellParams` with a `Lux` neural network surrogate `nn_sei`,
its state `st_sei`, and parameter `ComponentVector` axes `p_axes` for AD-compatible
Universal Differential Equation simulation and training.
"""
struct HybridSPMeTModel{C<:CellParams, N, S, A}
    cell::C
    nn_sei::N
    st_sei::S
    p_init::ComponentVector{Float64}
    p_axes::A
    j_scale::Float64
    enable_thermal::Bool
    enable_pore_clogging::Bool
end

"""
    HybridSPMeTModel(cell::CellParams = chen2020_params();
                     rng::AbstractRNG = Random.default_rng(),
                     hidden_dim::Int = 24,
                     init_scale::Real = 0.1,
                     j_scale::Real = J_DEG_SCALE,
                     enable_thermal::Bool = true,
                     enable_pore_clogging::Bool = true)

Construct a `HybridSPMeTModel` with initialized `Lux.Chain` SEI surrogate.
"""
function HybridSPMeTModel(
    cell::CellParams = chen2020_params();
    rng::AbstractRNG = Random.default_rng(),
    hidden_dim::Int = 24,
    init_scale::Real = 0.1,
    j_scale::Real = J_DEG_SCALE,
    enable_thermal::Bool = true,
    enable_pore_clogging::Bool = true,
)
    nn_sei, p_init, st_sei = init_sei_surrogate(rng; hidden_dim = hidden_dim, init_scale = init_scale)
    p_axes = getaxes(p_init)
    return HybridSPMeTModel(
        cell,
        nn_sei,
        st_sei,
        p_init,
        p_axes,
        Float64(j_scale),
        enable_thermal,
        enable_pore_clogging,
    )
end

"""
    ude_rhs(u::AbstractVector, p, t::Real, model::HybridSPMeTModel, I_app_fn)

Out-of-place (mutation-free, Zygote & ForwardDiff compatible) right-hand-side for the
45-state coupled SPMeT + Neural SEI Universal Differential Equation.
Returns `du::Vector{T}` where `T = promote_type(eltype(u), eltype(p))`.
"""
function ude_rhs(
    u::AbstractVector,
    p,
    t::Real,
    model::HybridSPMeTModel,
    I_app_fn,
)
    cell = model.cell
    p_eltype = (p === nothing || p isa SciMLBase.NullParameters) ? eltype(u) : eltype(p)
    T = promote_type(eltype(u), p_eltype, typeof(t), typeof(cell.R_n))

    N_r = cell.N_r
    N_x = num_electrolyte_cells(cell)

    c_sn      = @view u[1:N_r]
    c_sp      = @view u[(N_r + 1):(2 * N_r)]
    c_e_tilde = @view u[(2 * N_r + 1):(2 * N_r + N_x)]

    delta_sei_tilde = u[2 * N_r + N_x + 1]
    eps_e_n         = u[2 * N_r + N_x + 2]
    eps_s_n         = u[2 * N_r + N_x + 3]
    T_tilde         = u[2 * N_r + N_x + 5]

    T_kelvin = T(T_REF) + T(T_SCALE) * T_tilde

    theta_n_surf = surface_stoichiometry(c_sn, N_r)
    theta_p_surf = surface_stoichiometry(c_sp, N_r)

    c_e_n_bar, _, c_e_p_bar = domain_average_electrolyte(c_e_tilde, cell)

    a_sn = T(3.0) * max(eps_s_n, T(1.0e-4)) / T(cell.R_n)
    a_sp = T(3.0) * T(cell.eps_s_p) / T(cell.R_p)

    I_app   = T(I_app_fn(t))
    j_tot_n =  I_app / (T(cell.A_cell) * T(cell.L_n) * a_sn)
    j_tot_p = -I_app / (T(cell.A_cell) * T(cell.L_p) * a_sp)

    i0_n = exchange_current_density(cell.k_n, c_e_n_bar, cell.c_e0, cell.c_sn_max, theta_n_surf)
    i0_p = exchange_current_density(cell.k_p, c_e_p_bar, cell.c_e0, cell.c_sp_max, theta_p_surf)

    eta_int_n_approx = butler_volmer_overpotential(j_tot_n, i0_n, T_kelvin)

    # Evaluate parasitic SEI side-reaction current density j_deg <= 0
    j_deg = if p === nothing || p isa SciMLBase.NullParameters || model.j_scale == 0.0
        zero(T)
    else
        z_sei = sei_feature_vector(theta_n_surf, j_tot_n, eta_int_n_approx, delta_sei_tilde)
        eval_sei_current(model.nn_sei, z_sei, p, model.st_sei, model.p_axes; j_scale = model.j_scale)
    end

    j_int_n = j_tot_n - j_deg
    j_int_p = j_tot_p

    # 1. Solid-phase spherical FVM diffusion updates
    D_sn_T = solid_diffusivity(T(cell.D_sn), T(cell.E_a_dsn), T_kelvin, T(cell.T_ref))
    D_sp_T = solid_diffusivity(T(cell.D_sp), T(cell.E_a_dsp), T_kelvin, T(cell.T_ref))

    dc_sn = spherical_fvm_rhs(c_sn, D_sn_T, cell.R_n, j_int_n, cell.c_sn_max, N_r)
    dc_sp = spherical_fvm_rhs(c_sp, D_sp_T, cell.R_p, j_int_p, cell.c_sp_max, N_r)

    # 2. SEI growth and pore clogging (porosity rate feeds the electrolyte balance)
    ddelta_sei_tilde = -(T(cell.M_sei) / (T(2.0) * T(F_CONST) * T(cell.rho_sei) * T(DELTA_SEI_SCALE))) * j_deg
    deps_e_n         = model.enable_pore_clogging ? (-a_sn * T(DELTA_SEI_SCALE) * ddelta_sei_tilde) : zero(T)

    # 3. Electrolyte-phase axial FVM diffusion updates
    dc_e = electrolyte_fvm_rhs(c_e_tilde, eps_e_n, a_sn, a_sp, j_tot_n, j_tot_p, cell, deps_e_n)

    # 4. Remaining degradation & thermal ODEs
    deps_s_n         = zero(T)
    dQ_lli           = -(T(cell.A_cell) * T(cell.L_n) * a_sn / T(3600.0)) * j_deg

    dT_tilde = if model.enable_thermal
        eta_int_n = butler_volmer_overpotential(j_int_n, i0_n, T_kelvin)
        eta_int_p = butler_volmer_overpotential(j_int_p, i0_p, T_kelvin)
        U_n       = ocp_graphite(theta_n_surf)
        U_p       = ocp_nmc811(theta_p_surf)
        dphi_e    = electrolyte_overpotential(c_e_n_bar, c_e_p_bar, eps_e_n, I_app, T_kelvin, cell)
        R_film    = T(cell.R_film0) + (delta_sei_tilde * T(DELTA_SEI_SCALE)) / T(cell.kappa_sei)
        V_cell    = U_p - U_n + eta_int_p - eta_int_n + dphi_e - I_app * T(cell.R_ohm) - j_tot_n * R_film
        thermal_rhs(T_tilde, I_app, U_p, U_n, V_cell, cell)
    else
        zero(T)
    end

    return vcat(
        dc_sn,
        dc_sp,
        dc_e,
        [T(ddelta_sei_tilde), T(deps_e_n), T(deps_s_n), T(dQ_lli), T(dT_tilde)],
    )
end

"""
    spmet_rhs(u::AbstractVector, cell::CellParams, t::Real, I_app_fn;
              j_deg_fn = nothing, enable_thermal::Bool = true, enable_pore_clogging::Bool = true)

Out-of-place SPMeT RHS with either zero degradation (`j_deg_fn === nothing`) or an analytical
degradation law `j_deg_fn(theta_n_surf, eta_int_n_approx, delta_sei_tilde, T_kelvin)`.
"""
function spmet_rhs(
    u::AbstractVector,
    cell::CellParams,
    t::Real,
    I_app_fn;
    j_deg_fn = nothing,
    enable_thermal::Bool = true,
    enable_pore_clogging::Bool = true,
)
    T = promote_type(eltype(u), typeof(t), typeof(cell.R_n))
    N_r = cell.N_r
    N_x = num_electrolyte_cells(cell)

    c_sn      = @view u[1:N_r]
    c_sp      = @view u[(N_r + 1):(2 * N_r)]
    c_e_tilde = @view u[(2 * N_r + 1):(2 * N_r + N_x)]

    delta_sei_tilde = u[2 * N_r + N_x + 1]
    eps_e_n         = u[2 * N_r + N_x + 2]
    eps_s_n         = u[2 * N_r + N_x + 3]
    T_tilde         = u[2 * N_r + N_x + 5]

    T_kelvin = T(T_REF) + T(T_SCALE) * T_tilde

    theta_n_surf = surface_stoichiometry(c_sn, N_r)
    theta_p_surf = surface_stoichiometry(c_sp, N_r)

    c_e_n_bar, _, c_e_p_bar = domain_average_electrolyte(c_e_tilde, cell)

    a_sn = T(3.0) * max(eps_s_n, T(1.0e-4)) / T(cell.R_n)
    a_sp = T(3.0) * T(cell.eps_s_p) / T(cell.R_p)

    I_app   = T(I_app_fn(t))
    j_tot_n =  I_app / (T(cell.A_cell) * T(cell.L_n) * a_sn)
    j_tot_p = -I_app / (T(cell.A_cell) * T(cell.L_p) * a_sp)

    i0_n = exchange_current_density(cell.k_n, c_e_n_bar, cell.c_e0, cell.c_sn_max, theta_n_surf)
    i0_p = exchange_current_density(cell.k_p, c_e_p_bar, cell.c_e0, cell.c_sp_max, theta_p_surf)

    eta_int_n_approx = butler_volmer_overpotential(j_tot_n, i0_n, T_kelvin)

    j_deg = (j_deg_fn === nothing) ? zero(T) : T(j_deg_fn(theta_n_surf, eta_int_n_approx, delta_sei_tilde, T_kelvin))

    j_int_n = j_tot_n - j_deg
    j_int_p = j_tot_p

    D_sn_T = solid_diffusivity(T(cell.D_sn), T(cell.E_a_dsn), T_kelvin, T(cell.T_ref))
    D_sp_T = solid_diffusivity(T(cell.D_sp), T(cell.E_a_dsp), T_kelvin, T(cell.T_ref))

    dc_sn = spherical_fvm_rhs(c_sn, D_sn_T, cell.R_n, j_int_n, cell.c_sn_max, N_r)
    dc_sp = spherical_fvm_rhs(c_sp, D_sp_T, cell.R_p, j_int_p, cell.c_sp_max, N_r)
    ddelta_sei_tilde = -(T(cell.M_sei) / (T(2.0) * T(F_CONST) * T(cell.rho_sei) * T(DELTA_SEI_SCALE))) * j_deg
    deps_e_n         = enable_pore_clogging ? (-a_sn * T(DELTA_SEI_SCALE) * ddelta_sei_tilde) : zero(T)
    dc_e  = electrolyte_fvm_rhs(c_e_tilde, eps_e_n, a_sn, a_sp, j_tot_n, j_tot_p, cell, deps_e_n)
    deps_s_n         = zero(T)
    dQ_lli           = -(T(cell.A_cell) * T(cell.L_n) * a_sn / T(3600.0)) * j_deg

    dT_tilde = if enable_thermal
        eta_int_n = butler_volmer_overpotential(j_int_n, i0_n, T_kelvin)
        eta_int_p = butler_volmer_overpotential(j_int_p, i0_p, T_kelvin)
        U_n       = ocp_graphite(theta_n_surf)
        U_p       = ocp_nmc811(theta_p_surf)
        dphi_e    = electrolyte_overpotential(c_e_n_bar, c_e_p_bar, eps_e_n, I_app, T_kelvin, cell)
        R_film    = T(cell.R_film0) + (delta_sei_tilde * T(DELTA_SEI_SCALE)) / T(cell.kappa_sei)
        V_cell    = U_p - U_n + eta_int_p - eta_int_n + dphi_e - I_app * T(cell.R_ohm) - j_tot_n * R_film
        thermal_rhs(T_tilde, I_app, U_p, U_n, V_cell, cell)
    else
        zero(T)
    end

    return vcat(
        dc_sn,
        dc_sp,
        dc_e,
        [T(ddelta_sei_tilde), T(deps_e_n), T(deps_s_n), T(dQ_lli), T(dT_tilde)],
    )
end

"""
    voltage_cutoff_callback(cell::CellParams, I_app_fn; V_min::Real = 2.8)

Construct a `ContinuousCallback` that terminates integration when terminal voltage
`V_cell(u, I_app(t), cell)` drops down to `V_min` [V].
"""
function voltage_cutoff_callback(cell::CellParams, I_app_fn; V_min::Real = 2.8)
    condition(u, t, integrator) = terminal_voltage(u, I_app_fn(t), cell) - V_min
    affect!(integrator) = terminate!(integrator)
    return ContinuousCallback(condition, affect!; save_positions = (true, false))
end

"""
    build_ude_problem(model::HybridSPMeTModel, u0::AbstractVector, tspan::Tuple, I_app_fn;
                      p = model.p_init)

Construct an out-of-place `ODEProblem{false}` for the coupled SPMeT + Neural UDE system.
"""
function build_ude_problem(
    model::HybridSPMeTModel,
    u0::AbstractVector,
    tspan::Tuple,
    I_app_fn;
    p = model.p_init,
)
    f_oop(u, p_curr, t) = ude_rhs(u, p_curr, t, model, I_app_fn)
    return ODEProblem{false}(f_oop, u0, tspan, p)
end
