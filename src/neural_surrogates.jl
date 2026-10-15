# ============================================================================
# src/neural_surrogates.jl
# Lux.jl SEI & LAM Neural Surrogate Architectures and Thermodynamic Constraints
# ============================================================================

"""
    ad_softplus(x::Real)

Overflow-safe, smooth, AD-compatible `softplus(x) = log(1 + exp(x))`.
"""
@inline function ad_softplus(x::T) where {T<:Real}
    return ifelse(x > T(20.0), x, log1p(exp(x)))
end

"""
    build_sei_network(; hidden_dim::Int = 24)

Construct the Lux.jl neural network surrogate `nn_sei` for parasitic SEI reduction current:
`Dense(4 => 24, tanh) -> Dense(24 => 24, tanh) -> Dense(24 => 1)`.

Inputs `z_sei` (4-element normalized vector):
1. `theta_n_surf`
2. `j_tot_n / 10.0`
3. `(U_n(theta_n_surf) + eta_int_n_approx - 0.4) / 0.05`
4. `delta_sei_tilde`
"""
function build_sei_network(; hidden_dim::Int = 24)
    return Lux.Chain(
        Lux.Dense(4 => hidden_dim, tanh),
        Lux.Dense(hidden_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => 1),
    )
end

"""
    build_lam_network(; hidden_dim::Int = 16)

Construct the Lux.jl neural network surrogate `nn_lam` for mechanical-stress-driven
Loss of Active Material (LAM):
`Dense(2 => 16, tanh) -> Dense(16 => 16, tanh) -> Dense(16 => 1)`.

Inputs `z_lam` (2-element normalized vector):
1. `max_radial_grad_c_sn`
2. `T_tilde`
"""
function build_lam_network(; hidden_dim::Int = 16)
    return Lux.Chain(
        Lux.Dense(2 => hidden_dim, tanh),
        Lux.Dense(hidden_dim => hidden_dim, tanh),
        Lux.Dense(hidden_dim => 1),
    )
end

"""
    init_sei_surrogate(rng::AbstractRNG = Random.default_rng(); hidden_dim::Int = 24, init_scale::Real = 0.1)

Initialize `nn_sei`, returning `(nn_sei, p_init::ComponentVector{Float64}, st_sei)`.
Weights are scaled by `init_scale` so initial parasitic currents remain physically moderate.
"""
function init_sei_surrogate(
    rng::AbstractRNG = Random.default_rng();
    hidden_dim::Int = 24,
    init_scale::Real = 0.1,
)
    nn_sei = build_sei_network(; hidden_dim = hidden_dim)
    ps, st = Lux.setup(rng, nn_sei)
    p_vec  = ComponentVector{Float64}(ps) .* Float64(init_scale)
    # Bias the final layer slightly negative so initial j_deg is close to typical baseline
    p_vec.layer_3.bias .= -2.0
    return (nn_sei, p_vec, st)
end

"""
    sei_feature_vector(theta_n_surf::Real, j_tot_n::Real, eta_int_n_approx::Real, delta_sei_tilde::Real)

Build the 4-element normalized feature vector `z_sei`:
`[theta_n_surf, j_tot_n / 10.0, (U_n(theta_n_surf) + eta_int_n_approx - 0.4) / 0.05, delta_sei_tilde]`.
"""
@inline function sei_feature_vector(
    theta_n_surf::Real,
    j_tot_n::Real,
    eta_int_n_approx::Real,
    delta_sei_tilde::Real,
)
    T = promote_type(typeof(theta_n_surf), typeof(j_tot_n), typeof(eta_int_n_approx), typeof(delta_sei_tilde))
    U_n = ocp_graphite(T(theta_n_surf))
    eta_sei_norm = (U_n + T(eta_int_n_approx) - T(0.4)) / T(0.05)
    return [
        T(theta_n_surf),
        T(j_tot_n) / T(10.0),
        T(eta_sei_norm),
        T(delta_sei_tilde),
    ]
end

"""
    eval_sei_current(nn_sei, z_sei::AbstractVector, p, st_sei, p_axes = nothing; j_scale::Real = J_DEG_SCALE)

Evaluate the parasitic SEI current density `j_deg <= 0` [A / m^2] from `nn_sei`:
`j_deg = -j_scale * softplus(nn_out[1])`.
Supports both `ComponentVector` and flat `AbstractVector` parameters `p` (via `p_axes`).
"""
@inline function eval_sei_current(
    nn_sei,
    z_sei::AbstractVector,
    p,
    st_sei,
    p_axes = nothing;
    j_scale::Real = J_DEG_SCALE,
)
    T = promote_type(eltype(z_sei), eltype(p), typeof(j_scale))
    p_comp = (p isa ComponentVector || p_axes === nothing) ? p : ComponentVector(p, p_axes)
    nn_out, _ = Lux.apply(nn_sei, z_sei, p_comp, st_sei)
    return -T(j_scale) * ad_softplus(nn_out[1])
end

"""
    true_tafel_sei_current(theta_n_surf::Real, eta_int_n_approx::Real, delta_sei_tilde::Real,
                           T_kelvin::Real = T_REF; i0_sei::Real = 1.5e-6, U_sei_ref::Real = 0.4,
                           beta_sei::Real = 0.5, m_diff::Real = 1.0)

Ground-truth diffusion-limited cathodic Tafel kinetics for synthetic digital-twin generation:
`j_deg^true(t) = - (i0_sei * exp(-beta_sei * F / (R * T) * eta_sei(t))) / (1 + max(delta_sei_tilde(t), 0))^m_diff`,
where `eta_sei(t) = U_n(theta_n^surf) + eta_int_n_approx - U_sei_ref`. The defaults (`beta_sei = 0.5`,
`m_diff = 1`) give the standard symmetric Tafel-diffusion law; other values generate laws outside
the fixed-alpha SINDy library.
"""
@inline function true_tafel_sei_current(
    theta_n_surf::Real,
    eta_int_n_approx::Real,
    delta_sei_tilde::Real,
    T_kelvin::Real = T_REF;
    i0_sei::Real = 1.5e-6,
    U_sei_ref::Real = 0.4,
    beta_sei::Real = 0.5,
    m_diff::Real = 1.0,
)
    T = promote_type(typeof(theta_n_surf), typeof(eta_int_n_approx), typeof(delta_sei_tilde), typeof(T_kelvin), typeof(i0_sei))
    U_n = ocp_graphite(T(theta_n_surf))
    eta_sei = U_n + T(eta_int_n_approx) - T(U_sei_ref)
    # Clamp exponential argument to prevent numerical overflow during extreme trial steps
    exp_arg = clamp(-T(beta_sei) * T(F_CONST) / (T(R_GAS) * T(T_kelvin)) * eta_sei, T(-40.0), T(40.0))
    denom   = (one(T) + max(T(delta_sei_tilde), zero(T)))^T(m_diff)
    return -T(i0_sei) * exp(exp_arg) / denom
end
