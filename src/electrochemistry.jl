# ============================================================================
# src/electrochemistry.jl
# Open-Circuit Potentials, Butler-Volmer Kinetics, Electrolyte Overpotential,
# Terminal Voltage Observer, and Thermal Balance
# ============================================================================

"""
    ocp_graphite(theta_n::Real)

Smooth, differentiable Open-Circuit Potential `U_n(theta_n)` [V] for a Graphite anode:
`U_n(theta_n) = 0.12 + 1.5 * exp(-120 * theta_n) - 0.035 * tanh((theta_n - 0.55) / 0.08)`.
"""
@inline function ocp_graphite(theta_n::T) where {T<:Real}
    return T(0.12) + T(1.5) * exp(T(-120.0) * theta_n) - T(0.035) * tanh((theta_n - T(0.55)) / T(0.08))
end

"""
    ocp_nmc811(theta_p::Real)

Smooth, differentiable Open-Circuit Potential `U_p(theta_p)` [V] for an NMC811 cathode:
`U_p(theta_p) = 4.35 - 0.85 * theta_p - 0.15 * tanh((theta_p - 0.40) / 0.10) - 0.25 * exp(15 * (theta_p - 0.98))`.
"""
@inline function ocp_nmc811(theta_p::T) where {T<:Real}
    return T(4.35) - T(0.85) * theta_p - T(0.15) * tanh((theta_p - T(0.40)) / T(0.10)) - T(0.25) * exp(T(15.0) * (theta_p - T(0.98)))
end

"""
    exchange_current_density(k_k::Real, c_e_bar_tilde::Real, c_e0::Real, c_s_max::Real, theta_surf::Real)

Compute Butler-Volmer exchange current density `i_{0,k}` [A / m^2]:
`i_{0,k} = F * k_k * sqrt(c_e_bar_tilde * c_e0) * c_s_max * sqrt(theta_surf * (1 - theta_surf))`.
Domain clamping is applied to `c_e_bar_tilde` and `theta_surf` for AD safety.
"""
@inline function exchange_current_density(
    k_k::Real,
    c_e_bar_tilde::Real,
    c_e0::Real,
    c_s_max::Real,
    theta_surf::Real,
)
    T = promote_type(typeof(k_k), typeof(c_e_bar_tilde), typeof(c_e0), typeof(c_s_max), typeof(theta_surf))
    ce_safe = max(T(c_e_bar_tilde), T(1.0e-5))
    th_safe = clamp(T(theta_surf), T(1.0e-5), one(T) - T(1.0e-5))
    return T(F_CONST) * T(k_k) * sqrt(ce_safe * T(c_e0)) * T(c_s_max) * sqrt(th_safe * (one(T) - th_safe))
end

"""
    butler_volmer_overpotential(j_int::Real, i_0::Real, T_kelvin::Real)

Compute symmetric Butler-Volmer activation overpotential `eta_{int,k}` [V]:
`eta_{int,k} = (2 * R * T / F) * asinh(j_int / (2 * i_0))`.
"""
@inline function butler_volmer_overpotential(j_int::Real, i_0::Real, T_kelvin::Real)
    T = promote_type(typeof(j_int), typeof(i_0), typeof(T_kelvin))
    i0_safe = max(T(i_0), T(1.0e-12))
    return (T(2.0) * T(R_GAS) * T(T_kelvin) / T(F_CONST)) * asinh(T(j_int) / (T(2.0) * i0_safe))
end

"""
    electrolyte_overpotential(c_e_n_bar::Real, c_e_p_bar::Real, eps_e_n::Real,
                              I_app::Real, T_kelvin::Real, cell::CellParams)

Compute macroscopic electrolyte concentration and MacMullin ionic conduction overpotential `Delta_Phi_e(t)` [V]:
`Delta_Phi_e = (2*R*T*(1 - t_+^0)/F) * ln(c_e_p_bar / c_e_n_bar) - (I_app / (2 * A_cell)) * (L_n / (kappa_e * eps_e_n^1.5) + 2 * L_s / (kappa_e * eps_e_s^1.5) + L_p / (kappa_e * eps_e_p^1.5))`.
"""
@inline function electrolyte_overpotential(
    c_e_n_bar::Real,
    c_e_p_bar::Real,
    eps_e_n::Real,
    I_app::Real,
    T_kelvin::Real,
    cell::CellParams,
)
    T = promote_type(
        typeof(c_e_n_bar),
        typeof(c_e_p_bar),
        typeof(eps_e_n),
        typeof(I_app),
        typeof(T_kelvin),
        typeof(cell.kappa_e),
    )
    ce_n = max(T(c_e_n_bar), T(1.0e-5))
    ce_p = max(T(c_e_p_bar), T(1.0e-5))
    eps_n = max(T(eps_e_n), T(1.0e-4))
    eps_s = T(cell.eps_e_s)
    eps_p = T(cell.eps_e_p)
    ke    = T(cell.kappa_e)

    conc_term = (T(2.0) * T(R_GAS) * T(T_kelvin) * (one(T) - T(cell.t_plus)) / T(F_CONST)) * log(ce_p / ce_n)
    ohm_term  = (T(I_app) / (T(2.0) * T(cell.A_cell))) * (
        T(cell.L_n) / (ke * (eps_n ^ T(1.5))) +
        T(2.0) * T(cell.L_s) / (ke * (eps_s ^ T(1.5))) +
        T(cell.L_p) / (ke * (eps_p ^ T(1.5)))
    )
    return conc_term - ohm_term
end

"""
    terminal_voltage(u::AbstractVector, I_app::Real, cell::CellParams; j_deg::Real = 0.0)

Evaluate the algebraic cell terminal voltage `V_cell(t)` [V] from the dimensionless PDE state `u`,
applied current `I_app` [A] (`I_app > 0` for discharge, `I_app < 0` for charge), and optional
parasitic SEI current density `j_deg` [A / m^2]:

`V_cell(t) = U_p(theta_p^surf) - U_n(theta_n^surf) + eta_{int,p} - eta_{int,n} + Delta_Phi_e(t) - I_app * R_ohm - j_{tot,n} * (R_{film,0} + delta_sei_tilde * delta_scale / kappa_sei)`.
"""
function terminal_voltage(
    u::AbstractVector,
    I_app::Real,
    cell::CellParams;
    j_deg::Real = 0.0,
)
    T = promote_type(eltype(u), typeof(I_app), typeof(j_deg), typeof(cell.R_n))
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

    j_tot_n =  T(I_app) / (T(cell.A_cell) * T(cell.L_n) * a_sn)
    j_tot_p = -T(I_app) / (T(cell.A_cell) * T(cell.L_p) * a_sp)

    j_int_n = j_tot_n - T(j_deg)
    j_int_p = j_tot_p

    i0_n = exchange_current_density(cell.k_n, c_e_n_bar, cell.c_e0, cell.c_sn_max, theta_n_surf)
    i0_p = exchange_current_density(cell.k_p, c_e_p_bar, cell.c_e0, cell.c_sp_max, theta_p_surf)

    eta_int_n = butler_volmer_overpotential(j_int_n, i0_n, T_kelvin)
    eta_int_p = butler_volmer_overpotential(j_int_p, i0_p, T_kelvin)

    U_n = ocp_graphite(theta_n_surf)
    U_p = ocp_nmc811(theta_p_surf)

    dphi_e = electrolyte_overpotential(c_e_n_bar, c_e_p_bar, eps_e_n, I_app, T_kelvin, cell)
    R_film = T(cell.R_film0) + (delta_sei_tilde * T(DELTA_SEI_SCALE)) / T(cell.kappa_sei)

    V_cell = U_p - U_n + eta_int_p - eta_int_n + dphi_e - T(I_app) * T(cell.R_ohm) - j_tot_n * R_film
    return V_cell
end

"""
    thermal_rhs(T_tilde::Real, I_app::Real, U_p::Real, U_n::Real, V_cell::Real, cell::CellParams)

Compute dimensionless cell temperature derivative `d(T_tilde)/dt` where `T_tilde = (T - 298.15) / 10.0`:
`m_cell * C_p * dT/dt = I_app * (U_p - U_n - V_cell + T * d(U_p - U_n)/dT) - h_conv * A_surf * (T - T_amb)`.
"""
@inline function thermal_rhs(
    T_tilde::Real,
    I_app::Real,
    U_p::Real,
    U_n::Real,
    V_cell::Real,
    cell::CellParams,
)
    T = promote_type(typeof(T_tilde), typeof(I_app), typeof(U_p), typeof(U_n), typeof(V_cell), typeof(cell.m_cell))
    T_kelvin = T(T_REF) + T(T_SCALE) * T(T_tilde)
    dU_dT    = T(cell.dUp_dT) - T(cell.dUn_dT)

    q_irr_rev = T(I_app) * ((T(U_p) - T(U_n) - T(V_cell)) + T_kelvin * dU_dT)
    q_conv    = T(cell.h_conv) * T(cell.A_surf) * (T_kelvin - T(cell.T_amb))

    dT_dt = (q_irr_rev - q_conv) / (T(cell.m_cell) * T(cell.C_p))
    return dT_dt / T(T_SCALE)
end
