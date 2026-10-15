# ============================================================================
# src/fvm_discretization.jl
# Conservative Spherical Solid-Phase & Axial Electrolyte-Phase FVM Stencils
# ============================================================================

"""
    surface_stoichiometry(c_tilde::AbstractVector{T}, N_r::Int = length(c_tilde)) where {T}

Compute the particle surface stoichiometry `theta_k_surf` via second-order radial
extrapolation and clamp smoothly to `[1e-5, 1 - 1e-5]` for AD-safe square-root evaluation:
`theta_k^surf = clamp(1.5 * c_tilde[N_r] - 0.5 * c_tilde[N_r-1], 1e-5, 1 - 1e-5)`.
"""
@inline function surface_stoichiometry(c_tilde::AbstractVector{T}, N_r::Int = length(c_tilde)) where {T<:Real}
    theta_raw = T(1.5) * c_tilde[N_r] - T(0.5) * c_tilde[N_r - 1]
    lo = T(1.0e-5)
    hi = one(T) - T(1.0e-5)
    return clamp(theta_raw, lo, hi)
end

"""
    max_radial_concentration_gradient(c_tilde::AbstractVector{T}, N_r::Int = length(c_tilde)) where {T}

Compute the maximum dimensionless radial concentration gradient `max_i |c_{i+1} - c_i| / dr`
for active material mechanical stress (LAM) coupling.
"""
@inline function max_radial_concentration_gradient(c_tilde::AbstractVector{T}, N_r::Int = length(c_tilde)) where {T<:Real}
    dr = inv(T(N_r))
    max_grad = zero(T)
    for i in 1:(N_r - 1)
        g = abs(c_tilde[i + 1] - c_tilde[i]) / dr
        max_grad = ifelse(g > max_grad, g, max_grad)
    end
    return max_grad
end

"""
    spherical_fvm_rhs(c_tilde::AbstractVector, D_s, R_k, j_int, c_max, N_r::Int = length(c_tilde))

Out-of-place (Zygote-compatible, mutation-free) conservative spherical Finite-Volume
discretization for dimensionless solid lithium concentration `c_tilde = c_s / c_s_max`.

For shell `i in 1:N_r` with `dr = 1 / N_r`, `r_{i-1/2} = (i-1)*dr`, `r_{i+1/2} = i*dr`,
and `V_i = (r_{i+1/2}^3 - r_{i-1/2}^3) / 3`:
`dc_{s,k,i}/dt = (r_{i+1/2}^2 * F_{i+1/2} - r_{i-1/2}^2 * F_{i-1/2}) / V_i`.
"""
function spherical_fvm_rhs(
    c_tilde::AbstractVector,
    D_s::Real,
    R_k::Real,
    j_int::Real,
    c_max::Real,
    N_r::Int = length(c_tilde),
)
    T = promote_type(eltype(c_tilde), typeof(D_s), typeof(R_k), typeof(j_int), typeof(c_max))
    dr = inv(T(N_r))
    alpha = T(D_s) / (T(R_k)^2)
    bc_flux = -T(j_int) / (T(F_CONST) * T(c_max) * T(R_k))

    @inline function shell_rhs(i::Int)::T
        r_minus = T(i - 1) * dr
        r_plus  = T(i) * dr
        V_i     = (r_plus^3 - r_minus^3) / T(3)

        F_minus = (i == 1)   ? zero(T) : alpha * (c_tilde[i] - c_tilde[i - 1]) / dr
        F_plus  = (i == N_r) ? bc_flux : alpha * (c_tilde[i + 1] - c_tilde[i]) / dr

        return (r_plus^2 * F_plus - r_minus^2 * F_minus) / V_i
    end

    return map(shell_rhs, 1:N_r)
end

"""
    spherical_fvm_rhs!(dc_dt::AbstractVector, c_tilde::AbstractVector, D_s, R_k, j_int, c_max, N_r::Int = length(c_tilde))

In-place conservative spherical Finite-Volume discretization for dimensionless solid concentration.
"""
function spherical_fvm_rhs!(
    dc_dt::AbstractVector,
    c_tilde::AbstractVector,
    D_s::Real,
    R_k::Real,
    j_int::Real,
    c_max::Real,
    N_r::Int = length(c_tilde),
)
    T = promote_type(eltype(dc_dt), eltype(c_tilde), typeof(D_s), typeof(R_k), typeof(j_int), typeof(c_max))
    dr = inv(T(N_r))
    alpha = T(D_s) / (T(R_k)^2)
    bc_flux = -T(j_int) / (T(F_CONST) * T(c_max) * T(R_k))

    @inbounds for i in 1:N_r
        r_minus = T(i - 1) * dr
        r_plus  = T(i) * dr
        V_i     = (r_plus^3 - r_minus^3) / T(3)

        F_minus = (i == 1)   ? zero(T) : alpha * (c_tilde[i] - c_tilde[i - 1]) / dr
        F_plus  = (i == N_r) ? bc_flux : alpha * (c_tilde[i + 1] - c_tilde[i]) / dr

        dc_dt[i] = (r_plus^2 * F_plus - r_minus^2 * F_minus) / V_i
    end
    return dc_dt
end

"""
    electrolyte_fvm_rhs(c_e_tilde::AbstractVector, eps_e_n::Real, a_sn::Real, a_sp::Real,
                        j_tot_n::Real, j_tot_p::Real, cell::CellParams, deps_e_n_dt::Real = 0)

Out-of-place (Zygote-compatible, mutation-free) conservative axial Finite-Volume
discretization for dimensionless electrolyte concentration `c_e_tilde = c_e / c_e0`
across anode (`N_n`), separator (`N_s`), and cathode (`N_p`).

Uses Bruggeman effective diffusivity `D_{e,m}^eff = D_e * eps_{e,m}^1.5` and harmonic mean
diffusivity at control volume interfaces:
`D_{e,m+1/2}^eff = (dx_m + dx_{m+1}) / (dx_m / D_{e,m}^eff + dx_{m+1} / D_{e,m+1}^eff)`.

The anode porosity may change in time (SEI pore clogging). Lithium is conserved in the form
`d(eps_e * c_e)/dt = div(J) + source`, so anode cells carry the extra term `-c_e * deps_e_n_dt / eps_e_n`.
"""
function electrolyte_fvm_rhs(
    c_e_tilde::AbstractVector,
    eps_e_n::Real,
    a_sn::Real,
    a_sp::Real,
    j_tot_n::Real,
    j_tot_p::Real,
    cell::CellParams,
    deps_e_n_dt::Real = 0,
)
    T = promote_type(
        eltype(c_e_tilde),
        typeof(eps_e_n),
        typeof(deps_e_n_dt),
        typeof(a_sn),
        typeof(a_sp),
        typeof(j_tot_n),
        typeof(j_tot_p),
        typeof(cell.D_e),
    )

    N_n = cell.N_n
    N_s = cell.N_s
    N_p = cell.N_p
    N_x = N_n + N_s + N_p

    dx_n = T(cell.L_n) / T(N_n)
    dx_s = T(cell.L_s) / T(N_s)
    dx_p = T(cell.L_p) / T(N_p)

    eps_n_safe = max(T(eps_e_n), T(1.0e-4))
    eps_s_safe = T(cell.eps_e_s)
    eps_p_safe = T(cell.eps_e_p)

    De_n = T(cell.D_e) * (eps_n_safe ^ T(1.5))
    De_s = T(cell.D_e) * (eps_s_safe ^ T(1.5))
    De_p = T(cell.D_e) * (eps_p_safe ^ T(1.5))

    trans_factor = (one(T) - T(cell.t_plus)) / (T(F_CONST) * T(cell.c_e0))

    @inline function cell_props(m::Int)
        if m <= N_n
            return (dx_n, eps_n_safe, De_n, T(a_sn) * T(j_tot_n))
        elseif m <= N_n + N_s
            return (dx_s, eps_s_safe, De_s, zero(T))
        else
            return (dx_p, eps_p_safe, De_p, T(a_sp) * T(j_tot_p))
        end
    end

    @inline function interface_flux(m::Int)
        if m == 0 || m == N_x
            return zero(T)
        else
            dx_L, _, De_L, _ = cell_props(m)
            dx_R, _, De_R, _ = cell_props(m + 1)
            # Harmonic mean effective diffusivity:
            # D_{m+1/2}^eff = (dx_L + dx_R) / (dx_L/De_L + dx_R/De_R)
            # J_{m+1/2} = D_{m+1/2}^eff * (c_{m+1} - c_m) / (0.5 * (dx_L + dx_R))
            D_harm = (dx_L + dx_R) / (dx_L / De_L + dx_R / De_R)
            dist   = T(0.5) * (dx_L + dx_R)
            return D_harm * (c_e_tilde[m + 1] - c_e_tilde[m]) / dist
        end
    end

    @inline function cell_rhs(m::Int)::T
        dx_m, eps_m, _, aj_m = cell_props(m)
        J_plus  = interface_flux(m)
        J_minus = interface_flux(m - 1)
        dilution = m <= N_n ? -c_e_tilde[m] * T(deps_e_n_dt) / eps_m : zero(T)
        return (J_plus - J_minus) / (eps_m * dx_m) + (trans_factor / eps_m) * aj_m + dilution
    end

    return map(cell_rhs, 1:N_x)
end

"""
    electrolyte_fvm_rhs!(dc_e_dt::AbstractVector, c_e_tilde::AbstractVector, eps_e_n::Real,
                         a_sn::Real, a_sp::Real, j_tot_n::Real, j_tot_p::Real, cell::CellParams,
                         deps_e_n_dt::Real = 0)

In-place conservative axial Finite-Volume discretization for dimensionless electrolyte concentration.
"""
function electrolyte_fvm_rhs!(
    dc_e_dt::AbstractVector,
    c_e_tilde::AbstractVector,
    eps_e_n::Real,
    a_sn::Real,
    a_sp::Real,
    j_tot_n::Real,
    j_tot_p::Real,
    cell::CellParams,
    deps_e_n_dt::Real = 0,
)
    rhs = electrolyte_fvm_rhs(c_e_tilde, eps_e_n, a_sn, a_sp, j_tot_n, j_tot_p, cell, deps_e_n_dt)
    @inbounds for m in eachindex(rhs)
        dc_e_dt[m] = rhs[m]
    end
    return dc_e_dt
end

"""
    domain_average_electrolyte(c_e_tilde::AbstractVector{T}, cell::CellParams) where {T}

Compute domain-averaged dimensionless electrolyte concentrations `(c_e_n_bar, c_e_s_bar, c_e_p_bar)`,
clamped from below at `1e-5` for AD-safe logarithms and square roots.
"""
@inline function domain_average_electrolyte(c_e_tilde::AbstractVector{T}, cell::CellParams) where {T<:Real}
    N_n = cell.N_n
    N_s = cell.N_s
    N_p = cell.N_p

    sum_n = zero(T)
    for m in 1:N_n
        sum_n += c_e_tilde[m]
    end
    c_e_n_bar = max(sum_n / T(N_n), T(1.0e-5))

    sum_s = zero(T)
    for m in (N_n + 1):(N_n + N_s)
        sum_s += c_e_tilde[m]
    end
    c_e_s_bar = max(sum_s / T(N_s), T(1.0e-5))

    sum_p = zero(T)
    for m in (N_n + N_s + 1):(N_n + N_s + N_p)
        sum_p += c_e_tilde[m]
    end
    c_e_p_bar = max(sum_p / T(N_p), T(1.0e-5))

    return (c_e_n_bar, c_e_s_bar, c_e_p_bar)
end

"""
    solid_lithium_inventory(u::AbstractVector{T}, cell::CellParams) where {T}

Compute total moles of lithium `N_Li^solid` [mol] stored across both solid electrodes:
`N_Li^solid = A_cell * (L_n * eps_s_n * c_sn_max * theta_n_avg + L_p * eps_s_p * c_sp_max * theta_p_avg)`,
where `theta_k_avg = 3 * sum_i V_i * c_tilde_{s,k,i}`.
"""
function solid_lithium_inventory(u::AbstractVector{T}, cell::CellParams) where {T<:Real}
    N_r = cell.N_r
    N_x = num_electrolyte_cells(cell)
    dr  = inv(T(N_r))

    c_sn = @view u[1:N_r]
    c_sp = @view u[(N_r + 1):(2 * N_r)]
    eps_s_n = u[2 * N_r + N_x + 3]

    theta_n_avg = zero(T)
    theta_p_avg = zero(T)
    for i in 1:N_r
        r_minus = T(i - 1) * dr
        r_plus  = T(i) * dr
        V_i     = (r_plus^3 - r_minus^3) / T(3)
        theta_n_avg += T(3) * V_i * c_sn[i]
        theta_p_avg += T(3) * V_i * c_sp[i]
    end

    n_li_n = T(cell.A_cell) * T(cell.L_n) * eps_s_n * T(cell.c_sn_max) * theta_n_avg
    n_li_p = T(cell.A_cell) * T(cell.L_p) * T(cell.eps_s_p) * T(cell.c_sp_max) * theta_p_avg
    return n_li_n + n_li_p
end

"""
    electrolyte_lithium_inventory(u::AbstractVector{T}, cell::CellParams) where {T}

Compute total moles of lithium ions `N_Li^elec` [mol] dissolved in the electrolyte phase
across the anode, separator, and cathode control volumes.
"""
function electrolyte_lithium_inventory(u::AbstractVector{T}, cell::CellParams) where {T<:Real}
    N_r = cell.N_r
    N_n = cell.N_n
    N_s = cell.N_s
    N_p = cell.N_p
    N_x = N_n + N_s + N_p

    c_e_tilde = @view u[(2 * N_r + 1):(2 * N_r + N_x)]
    eps_e_n   = u[2 * N_r + N_x + 2]

    dx_n = T(cell.L_n) / T(N_n)
    dx_s = T(cell.L_s) / T(N_s)
    dx_p = T(cell.L_p) / T(N_p)

    moles_per_area = zero(T)
    for m in 1:N_n
        moles_per_area += eps_e_n * dx_n * c_e_tilde[m]
    end
    for m in (N_n + 1):(N_n + N_s)
        moles_per_area += T(cell.eps_e_s) * dx_s * c_e_tilde[m]
    end
    for m in (N_n + N_s + 1):N_x
        moles_per_area += T(cell.eps_e_p) * dx_p * c_e_tilde[m]
    end

    return T(cell.A_cell) * T(cell.c_e0) * moles_per_area
end
