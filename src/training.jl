# ============================================================================
# src/training.jl
# Adjoint Loss Functions, Synthetic Digital Twin Generator (variable C-rate
# accelerated cycles), Observable-Only Rollout Training (Adam + L-BFGS),
# and Ground-Truth Evaluation Helpers
# ============================================================================

"""
    CycleRecord

One cycle of synthetic data.

Observable fields (what a cycler measures, and the only data used for training):
`t_grid`, `I_amp`, `I_grid`, `V_noisy`, `Q_lli_meas_end` (noisy cumulative capacity fade at the end of the cycle).

Ground-truth fields (hidden states of the generating model, used only for evaluation and plots):
`V_true`, `u0`, `uf`, `delta_sei_*`, `Q_lli_start/end`, and the `*_samples` vectors.
"""
struct CycleRecord
    cycle_idx::Int
    t_grid::Vector{Float64}
    I_amp::Float64
    I_grid::Vector{Float64}
    V_true::Vector{Float64}
    V_noisy::Vector{Float64}
    u0::Vector{Float64}
    uf::Vector{Float64}
    delta_sei_start::Float64
    delta_sei_end::Float64
    Q_lli_start::Float64
    Q_lli_end::Float64
    Q_lli_meas_end::Float64
    theta_n_samples::Vector{Float64}
    j_tot_n_samples::Vector{Float64}
    eta_n_samples::Vector{Float64}
    delta_sei_samples::Vector{Float64}
    j_deg_samples::Vector{Float64}
end

"""
    SyntheticTwinDataset

Container holding multi-cycle synthetic data across `N_cycles` accelerated aging cycles,
split into training and holdout cycle indices, plus the parameters of the hidden SEI law.
"""
struct SyntheticTwinDataset
    cell::CellParams{Float64}
    cycles::Vector{CycleRecord}
    train_indices::Vector{Int}
    holdout_indices::Vector{Int}
    i0_sei_true::Float64
    U_sei_ref::Float64
    accel_factor::Float64
    noise_std_V::Float64
    noise_std_Q::Float64
    beta_sei::Float64
    m_diff::Float64
end

"""
    default_cycle_current(t::Real, I_amp::Real = 5.0, period::Real = 120.0)

Smooth dynamic charge/discharge current profile `I_app(t)` [A] that sweeps anode surface
stoichiometry and overpotential across both charging (`I < 0`) and discharging (`I > 0`) regimes.
"""
@inline function default_cycle_current(t::Real, I_amp::Real = 5.0, period::Real = 120.0)
    T = promote_type(typeof(t), typeof(I_amp), typeof(period))
    # Asymmetric CC-like harmonic profile with net charge-sustaining swing
    return T(I_amp) * ( -T(0.65) + T(0.85) * cos(T(2.0 * pi) * T(t) / T(period)) )
end

"""
    default_c_rate_schedule(N_cycles::Int, train_cycles::Int)

C-rate multiplier for each cycle. Training cycles rotate through `(0.6, 1.0, 1.4)` so the
per-cycle capacity fade carries information about the overpotential dependence of the SEI rate.
Holdout cycles rotate through `(0.8, 1.8)`: one rate inside the training range and one outside it,
neither of which appears during training.
"""
function default_c_rate_schedule(N_cycles::Int, train_cycles::Int)
    train_rates   = (0.6, 1.0, 1.4)
    holdout_rates = (0.8, 1.8)
    return [c <= train_cycles ? train_rates[mod1(c, 3)] : holdout_rates[mod1(c - train_cycles, 2)] for c in 1:N_cycles]
end

"""
    adjoint_trajectory_loss(p, model::HybridSPMeTModel, u0::AbstractVector,
                            t_save::AbstractVector, I_app_fn,
                            V_target::AbstractVector, Q_lli_target::Real;
                            w_V::Real = 1.0, w_Q::Real = 100.0,
                            sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
                            solver = Rodas5P(),
                            reltol::Real = 1e-6, abstol::Real = 1e-8)

Multi-objective scalar loss over an ODE trajectory comparing simulated terminal voltage
`V_cell(t_k; p)` and final lost lithium inventory `Q_lli(t_f; p)` against targets.
Fully compatible with both `ForwardDiff.gradient` and `Zygote.gradient` via `SciMLSensitivity`.
"""
function adjoint_trajectory_loss(
    p,
    model::HybridSPMeTModel,
    u0::AbstractVector,
    t_save::AbstractVector,
    I_app_fn,
    V_target::AbstractVector,
    Q_lli_target::Real;
    w_V::Real = 1.0,
    w_Q::Real = 100.0,
    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
    solver = Rodas5P(),
    reltol::Real = 1.0e-6,
    abstol::Real = 1.0e-8,
)
    T = promote_type(eltype(p), eltype(u0))
    cell = model.cell
    N_r  = cell.N_r
    N_x  = num_electrolyte_cells(cell)
    idx_Q_lli = 2 * N_r + N_x + 4

    tspan = (Float64(first(t_save)), Float64(last(t_save)))
    u0_T  = T.(u0)

    f_oop(u, p_curr, t) = ude_rhs(u, p_curr, t, model, I_app_fn)
    prob = ODEProblem{false}(f_oop, u0_T, tspan, p)

    sol = solve(
        prob,
        solver;
        u0 = u0_T,
        p = p,
        saveat = t_save,
        sensealg = sensealg,
        reltol = reltol,
        abstol = abstol,
    )

    # Array(sol) has shape (D, length(t_save)) and is seamlessly tracked by both Zygote and ForwardDiff
    U_mat = Array(sol)
    n_pts = size(U_mat, 2)

    loss_V = zero(T)
    for k in 1:n_pts
        u_k   = @view U_mat[:, k]
        t_k   = t_save[k]
        I_k   = I_app_fn(t_k)

        # Compute j_deg consistent with u_k and p
        c_sn      = @view u_k[1:N_r]
        c_e_tilde = @view u_k[(2 * N_r + 1):(2 * N_r + N_x)]
        delta_sei = u_k[2 * N_r + N_x + 1]
        eps_s_n   = u_k[2 * N_r + N_x + 3]
        T_tilde   = u_k[2 * N_r + N_x + 5]
        T_kelvin  = T(T_REF) + T(T_SCALE) * T_tilde

        theta_n_surf = surface_stoichiometry(c_sn, N_r)
        c_e_n_bar, _, _ = domain_average_electrolyte(c_e_tilde, cell)
        a_sn    = T(3.0) * max(eps_s_n, T(1.0e-4)) / T(cell.R_n)
        j_tot_n = T(I_k) / (T(cell.A_cell) * T(cell.L_n) * a_sn)
        i0_n    = exchange_current_density(cell.k_n, c_e_n_bar, cell.c_e0, cell.c_sn_max, theta_n_surf)
        eta_n   = butler_volmer_overpotential(j_tot_n, i0_n, T_kelvin)
        z_sei   = sei_feature_vector(theta_n_surf, j_tot_n, eta_n, delta_sei)
        j_deg_k = eval_sei_current(model.nn_sei, z_sei, p, model.st_sei, model.p_axes; j_scale = model.j_scale)

        V_sim_k = terminal_voltage(u_k, I_k, cell; j_deg = j_deg_k)
        loss_V += (V_sim_k - T(V_target[k]))^2
    end
    loss_V /= T(n_pts)

    Q_lli_final = U_mat[idx_Q_lli, end]
    loss_Q = (Q_lli_final - T(Q_lli_target))^2

    return T(w_V) * loss_V + T(w_Q) * loss_Q
end

"""
    cycle_start_state(cell::CellParams, delta_sei, eps_e_n, Q_lli)

Initial state of a cycle under the shared test protocol: after a rest, the electrolyte is
uniform, the temperature is ambient and the electrodes start at a reference stoichiometry
shifted by the lithium already lost. Only the aging states `(delta_sei, eps_e_n, Q_lli)` are
carried from the previous cycle. Mutation-free so Zygote can differentiate through it.
"""
function cycle_start_state(cell::CellParams, delta_sei, eps_e_n, Q_lli)
    N_r = cell.N_r
    N_x = num_electrolyte_cells(cell)
    T = promote_type(typeof(delta_sei), typeof(eps_e_n), typeof(Q_lli))
    total_li_n_cap_Ah = (F_CONST * cell.A_cell * cell.L_n * cell.eps_s_n0 * cell.c_sn_max) / 3600.0
    theta_n = clamp(T(0.80) - T(0.5) * Q_lli / T(total_li_n_cap_Ah), T(0.25), T(0.90))
    theta_p = clamp(T(0.30) + T(0.5) * Q_lli / T(total_li_n_cap_Ah), T(0.15), T(0.85))
    return vcat(
        fill(theta_n, N_r),
        fill(theta_p, N_r),
        ones(T, N_x),
        [T(delta_sei), T(eps_e_n), T(cell.eps_s_n0), T(Q_lli), zero(T)],
    )
end

"""
    sei_inputs(u::AbstractVector, I_app::Real, cell::CellParams)

Return `(theta_n_surf, j_tot_n, eta_int_n_approx, delta_sei_tilde, T_kelvin)` at state `u`.
"""
function sei_inputs(u::AbstractVector, I_app::Real, cell::CellParams)
    N_r = cell.N_r
    N_x = num_electrolyte_cells(cell)
    T = promote_type(eltype(u), typeof(I_app))
    c_sn      = @view u[1:N_r]
    c_e_tilde = @view u[(2 * N_r + 1):(2 * N_r + N_x)]
    delta_sei = u[2 * N_r + N_x + 1]
    eps_s_n   = u[2 * N_r + N_x + 3]
    T_kelvin  = T(T_REF) + T(T_SCALE) * u[2 * N_r + N_x + 5]

    theta_n_surf = surface_stoichiometry(c_sn, N_r)
    c_e_n_bar, _, _ = domain_average_electrolyte(c_e_tilde, cell)
    a_sn    = T(3.0) * max(eps_s_n, T(1.0e-4)) / T(cell.R_n)
    j_tot_n = T(I_app) / (T(cell.A_cell) * T(cell.L_n) * a_sn)
    i0_n    = exchange_current_density(cell.k_n, c_e_n_bar, cell.c_e0, cell.c_sn_max, theta_n_surf)
    eta_n   = butler_volmer_overpotential(j_tot_n, i0_n, T_kelvin)
    return (theta_n_surf, j_tot_n, eta_n, delta_sei, T_kelvin)
end

"""
    predicted_voltage(model::HybridSPMeTModel, p, u::AbstractVector, I_app::Real)

Terminal voltage of the hybrid model at state `u`, with the network's SEI current.
"""
function predicted_voltage(model::HybridSPMeTModel, p, u::AbstractVector, I_app::Real)
    th_n, j_tot_n, eta_n, d_sei, _ = sei_inputs(u, I_app, model.cell)
    z_sei = sei_feature_vector(th_n, j_tot_n, eta_n, d_sei)
    j_deg = eval_sei_current(model.nn_sei, z_sei, p, model.st_sei, model.p_axes; j_scale = model.j_scale)
    return terminal_voltage(u, I_app, model.cell; j_deg = j_deg)
end

"""
    generate_synthetic_twin_data(cell::CellParams = chen2020_params();
                                 N_cycles::Int = 25,
                                 train_cycles::Int = 18,
                                 t_cycle::Real = 120.0,
                                 dt_save::Real = 10.0,
                                 c_rates = default_c_rate_schedule(N_cycles, train_cycles),
                                 i0_sei_true::Real = 2.0e-6,
                                 U_sei_ref::Real = 0.40,
                                 accel_factor::Real = 150.0,
                                 beta_sei::Real = 0.5,
                                 m_diff::Real = 1.0,
                                 noise_std_V::Real = 1.0e-3,
                                 noise_std_Q::Real = 1.0e-3,
                                 rng::AbstractRNG = Random.MersenneTwister(42))

Generate a multi-cycle synthetic dataset from the SPMeT driven by the hidden SEI law
`j_deg = -i0_eff * exp(-beta_sei * F / (R * T) * eta_sei) / (1 + delta_sei_tilde)^m_diff`
with `i0_eff = i0_sei_true * accel_factor`. Cycle `c` uses current amplitude `c_rates[c] * I_1C`.
Each cycle records noisy terminal voltage (`noise_std_V`, V) and noisy cumulative capacity fade
(`noise_std_Q`, Ah); with active-material loss disabled the capacity fade equals the lost lithium inventory.
"""
function generate_synthetic_twin_data(
    cell::CellParams = chen2020_params();
    N_cycles::Int = 25,
    train_cycles::Int = 18,
    t_cycle::Real = 120.0,
    dt_save::Real = 10.0,
    c_rates = default_c_rate_schedule(N_cycles, train_cycles),
    i0_sei_true::Real = 2.0e-6,
    U_sei_ref::Real = 0.40,
    accel_factor::Real = 150.0,
    beta_sei::Real = 0.5,
    m_diff::Real = 1.0,
    noise_std_V::Real = 1.0e-3,
    noise_std_Q::Real = 1.0e-3,
    rng::AbstractRNG = Random.MersenneTwister(42),
)
    length(c_rates) == N_cycles || error("c_rates must have one entry per cycle")
    N_r = cell.N_r
    N_x = num_electrolyte_cells(cell)
    idx_delta = 2 * N_r + N_x + 1
    idx_eps_e = 2 * N_r + N_x + 2
    idx_Q_lli = 2 * N_r + N_x + 4

    t_grid = collect(0.0:Float64(dt_save):Float64(t_cycle))

    i0_eff = Float64(i0_sei_true) * Float64(accel_factor)
    true_law(theta_n_surf, eta_int_n_approx, delta_sei_tilde, T_kelvin) =
        true_tafel_sei_current(
            theta_n_surf,
            eta_int_n_approx,
            delta_sei_tilde,
            T_kelvin;
            i0_sei = i0_eff,
            U_sei_ref = U_sei_ref,
            beta_sei = beta_sei,
            m_diff = m_diff,
        )

    cycles = Vector{CycleRecord}(undef, N_cycles)

    delta_sei_curr = 0.0
    eps_e_n_curr   = cell.eps_e_n0
    Q_lli_curr     = 0.0

    for c in 1:N_cycles
        I_amp = Float64(c_rates[c]) * cell.I_1C
        I_app_fn(t) = default_cycle_current(t, I_amp, t_cycle)
        I_grid = [I_app_fn(t) for t in t_grid]

        u0_c = cycle_start_state(cell, delta_sei_curr, eps_e_n_curr, Q_lli_curr)

        f_true(u, p_null, t) = spmet_rhs(
            u,
            cell,
            t,
            I_app_fn;
            j_deg_fn = true_law,
            enable_thermal = true,
            enable_pore_clogging = true,
        )
        prob_c = ODEProblem{false}(f_true, u0_c, (first(t_grid), last(t_grid)))
        sol_c  = solve(prob_c, Rodas5P(); saveat = t_grid, reltol = 1e-6, abstol = 1e-8)

        n_pts = length(t_grid)
        V_true  = zeros(Float64, n_pts)
        V_noisy = zeros(Float64, n_pts)
        theta_n_samples   = zeros(Float64, n_pts)
        j_tot_n_samples   = zeros(Float64, n_pts)
        eta_n_samples     = zeros(Float64, n_pts)
        delta_sei_samples = zeros(Float64, n_pts)
        j_deg_samples     = zeros(Float64, n_pts)

        for k in 1:n_pts
            u_k = sol_c.u[k]
            th_n, j_tot_n, eta_n, d_sei_k, T_kelvin = sei_inputs(u_k, I_grid[k], cell)
            j_deg_k = true_law(th_n, eta_n, d_sei_k, T_kelvin)

            V_k = terminal_voltage(u_k, I_grid[k], cell; j_deg = j_deg_k)
            V_true[k]  = V_k
            V_noisy[k] = V_k + Float64(noise_std_V) * randn(rng)

            theta_n_samples[k]   = th_n
            j_tot_n_samples[k]   = j_tot_n
            eta_n_samples[k]     = eta_n
            delta_sei_samples[k] = d_sei_k
            j_deg_samples[k]     = j_deg_k
        end

        uf_c = copy(sol_c.u[end])
        cycles[c] = CycleRecord(
            c,
            t_grid,
            I_amp,
            I_grid,
            V_true,
            V_noisy,
            u0_c,
            uf_c,
            u0_c[idx_delta],
            uf_c[idx_delta],
            u0_c[idx_Q_lli],
            uf_c[idx_Q_lli],
            uf_c[idx_Q_lli] + Float64(noise_std_Q) * randn(rng),
            theta_n_samples,
            j_tot_n_samples,
            eta_n_samples,
            delta_sei_samples,
            j_deg_samples,
        )

        # Carry the aging states into cycle c + 1
        delta_sei_curr = uf_c[idx_delta]
        eps_e_n_curr   = uf_c[idx_eps_e]
        Q_lli_curr     = uf_c[idx_Q_lli]
    end

    return SyntheticTwinDataset(
        cell,
        cycles,
        collect(1:train_cycles),
        collect((train_cycles + 1):N_cycles),
        i0_eff,
        Float64(U_sei_ref),
        Float64(accel_factor),
        Float64(noise_std_V),
        Float64(noise_std_Q),
        Float64(beta_sei),
        Float64(m_diff),
    )
end

"""
    rollout_cycles(model::HybridSPMeTModel, p, dataset::SyntheticTwinDataset; last_cycle::Int = length(dataset.cycles),
                   solver = Rodas5P(), reltol::Real = 1e-6, abstol::Real = 1e-8)

Roll the hybrid model forward from a fresh cell through cycles `1:last_cycle`, using only the
known protocol (current amplitude per cycle) and the model's own predicted aging states.
Returns a vector of named tuples `(; V_pred, Q_lli_end, states)` per cycle, where `states`
holds the predicted solution at each saved time.
"""
function rollout_cycles(
    model::HybridSPMeTModel,
    p,
    dataset::SyntheticTwinDataset;
    last_cycle::Int = length(dataset.cycles),
    solver = Rodas5P(),
    reltol::Real = 1.0e-6,
    abstol::Real = 1.0e-8,
)
    cell = model.cell
    N_r  = cell.N_r
    N_x  = num_electrolyte_cells(cell)
    idx_delta = 2 * N_r + N_x + 1
    idx_eps_e = 2 * N_r + N_x + 2
    idx_Q_lli = 2 * N_r + N_x + 4

    delta_curr = 0.0
    eps_e_curr = cell.eps_e_n0
    Q_curr     = 0.0

    out = Vector{Any}(undef, last_cycle)
    for c in 1:last_cycle
        cyc = dataset.cycles[c]
        I_app_fn(t) = default_cycle_current(t, cyc.I_amp, last(cyc.t_grid))
        u0   = cycle_start_state(cell, delta_curr, eps_e_curr, Q_curr)
        prob = build_ude_problem(model, u0, (first(cyc.t_grid), last(cyc.t_grid)), I_app_fn; p = p)
        sol  = solve(prob, solver; saveat = cyc.t_grid, reltol = reltol, abstol = abstol)

        V_pred = [predicted_voltage(model, p, sol.u[k], cyc.I_grid[k]) for k in eachindex(cyc.t_grid)]
        uf = sol.u[end]
        out[c] = (; V_pred, Q_lli_end = uf[idx_Q_lli], states = sol.u)

        delta_curr = uf[idx_delta]
        eps_e_curr = uf[idx_eps_e]
        Q_curr     = uf[idx_Q_lli]
    end
    return out
end

"""
    evaluate_holdout_cycles(model::HybridSPMeTModel, p, dataset::SyntheticTwinDataset;
                            indices = dataset.holdout_indices)

Roll out the model from a fresh cell through `1:maximum(indices)` and compute voltage RMSE
(against the noise-free voltage) and capacity-fade RMSE on the requested cycles.
"""
function evaluate_holdout_cycles(
    model::HybridSPMeTModel,
    p,
    dataset::SyntheticTwinDataset;
    indices = dataset.holdout_indices,
)
    roll = rollout_cycles(model, p, dataset; last_cycle = maximum(indices))
    sq_V = 0.0
    n_V  = 0
    sq_Q = 0.0
    for c in indices
        cyc = dataset.cycles[c]
        sq_V += sum(abs2, roll[c].V_pred .- cyc.V_true)
        n_V  += length(cyc.V_true)
        sq_Q += (roll[c].Q_lli_end - cyc.Q_lli_end)^2
    end
    voltage_rmse_V   = sqrt(sq_V / max(n_V, 1))
    capacity_rmse_Ah = sqrt(sq_Q / max(length(indices), 1))
    return (;
        voltage_rmse_V,
        voltage_rmse_mV = voltage_rmse_V * 1000.0,
        capacity_rmse_Ah,
    )
end

"""
    sei_rate_parity(model::HybridSPMeTModel, p, dataset::SyntheticTwinDataset;
                    indices = eachindex(dataset.cycles))

Compare the network's SEI current with the hidden true SEI current at the true states visited
in the requested cycles. This uses ground truth and is for evaluation only.
Returns `(; j_true, j_pred, r2, rel_rmse)`.
"""
function sei_rate_parity(
    model::HybridSPMeTModel,
    p,
    dataset::SyntheticTwinDataset;
    indices = eachindex(dataset.cycles),
)
    j_true = Float64[]
    j_pred = Float64[]
    for c in indices
        cyc = dataset.cycles[c]
        for k in eachindex(cyc.t_grid)
            z = sei_feature_vector(cyc.theta_n_samples[k], cyc.j_tot_n_samples[k], cyc.eta_n_samples[k], cyc.delta_sei_samples[k])
            push!(j_pred, eval_sei_current(model.nn_sei, z, p, model.st_sei, model.p_axes; j_scale = model.j_scale))
            push!(j_true, cyc.j_deg_samples[k])
        end
    end
    r2 = 1.0 - sum(abs2, j_true .- j_pred) / sum(abs2, j_true .- mean(j_true))
    rel_rmse = sqrt(mean(abs2, j_true .- j_pred)) / sqrt(mean(abs2, j_true))
    return (; j_true, j_pred, r2, rel_rmse)
end

"""
    observable_rollout_loss(p, model::HybridSPMeTModel, dataset::SyntheticTwinDataset;
                            sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
                            solver = Rodas5P(), reltol::Real = 1e-6, abstol::Real = 1e-8,
                            reg_weight::Real = 1e-6)

Training loss that uses only measurable quantities. The hybrid model is rolled out from a fresh
cell through all training cycles (which must be `1:n`), carrying its own predicted aging states
between cycles. Each cycle contributes the noise-normalized voltage misfit
`mean((V_pred - V_noisy)^2) / noise_std_V^2` and capacity misfit
`((Q_pred_end - Q_meas_end) / noise_std_Q)^2`. No hidden state or SEI current enters the loss.
"""
function observable_rollout_loss(
    p,
    model::HybridSPMeTModel,
    dataset::SyntheticTwinDataset;
    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
    solver = Rodas5P(),
    reltol::Real = 1.0e-6,
    abstol::Real = 1.0e-8,
    reg_weight::Real = 1.0e-6,
)
    T = eltype(p)
    cell = model.cell
    N_r  = cell.N_r
    N_x  = num_electrolyte_cells(cell)
    idx_delta = 2 * N_r + N_x + 1
    idx_eps_e = 2 * N_r + N_x + 2
    idx_Q_lli = 2 * N_r + N_x + 4

    train = dataset.train_indices
    train == collect(1:length(train)) || error("training cycles must be 1:n so the rollout starts from a fresh cell")

    sigma_V = T(dataset.noise_std_V)
    sigma_Q = T(dataset.noise_std_Q)

    delta_curr = zero(T)
    eps_e_curr = T(cell.eps_e_n0)
    Q_curr     = zero(T)

    total = zero(T)
    for c in train
        cyc = dataset.cycles[c]
        I_amp = cyc.I_amp
        t_end = last(cyc.t_grid)
        I_app_fn(t) = default_cycle_current(t, I_amp, t_end)

        u0 = cycle_start_state(cell, delta_curr, eps_e_curr, Q_curr)
        f_oop(u, p_curr, t) = ude_rhs(u, p_curr, t, model, I_app_fn)
        prob = ODEProblem{false}(f_oop, u0, (first(cyc.t_grid), t_end), p)
        sol  = solve(prob, solver; u0 = u0, p = p, saveat = cyc.t_grid, sensealg = sensealg, reltol = reltol, abstol = abstol)
        U = Array(sol)

        n_pts = size(U, 2)
        loss_V = zero(T)
        for k in 1:n_pts
            V_k = predicted_voltage(model, p, U[:, k], cyc.I_grid[k])
            loss_V += ((V_k - T(cyc.V_noisy[k])) / sigma_V)^2
        end
        Q_end  = U[idx_Q_lli, end]
        loss_Q = ((Q_end - T(cyc.Q_lli_meas_end)) / sigma_Q)^2
        total += loss_V / T(n_pts) + loss_Q

        delta_curr = U[idx_delta, end]
        eps_e_curr = U[idx_eps_e, end]
        Q_curr     = Q_end
    end

    return total / T(length(train)) + T(reg_weight) * sum(abs2, p)
end

"""
    threaded_forward_gradient(f, p::AbstractVector; n_blocks::Int = Threads.nthreads())

Gradient of the scalar function `f` by forward-mode AD, with the parameters split into
`n_blocks` contiguous blocks that are differentiated on separate threads. For the small
networks used here, forward mode through the stiff solver is faster than the adjoint
(no adjoint ODE, no Jacobian of the augmented system), and the blocks parallelize perfectly.
"""
function threaded_forward_gradient(f, p::AbstractVector; n_blocks::Int = Threads.nthreads())
    p_flat = collect(Float64, p)
    n = length(p_flat)
    n_blocks = clamp(n_blocks, 1, n)
    bounds = round.(Int, range(0, n, length = n_blocks + 1))
    blocks = [(bounds[b] + 1):bounds[b + 1] for b in 1:n_blocks]
    parts = Vector{Vector{Float64}}(undef, n_blocks)
    Threads.@sync for b in 1:n_blocks
        Threads.@spawn begin
            idx = blocks[b]
            function f_block(x)
                q = convert(Vector{eltype(x)}, p_flat)
                q[idx] = x
                return f(q)
            end
            cfg = ForwardDiff.GradientConfig(f_block, p_flat[idx], ForwardDiff.Chunk(min(length(idx), 12)))
            parts[b] = ForwardDiff.gradient(f_block, p_flat[idx], cfg)
        end
    end
    return reduce(vcat, parts)
end

"""
    train_battery_ude!(model::HybridSPMeTModel, dataset::SyntheticTwinDataset;
                       adam_lr::Real = 5e-3,
                       adam_iters::Int = 100,
                       lbfgs_iters::Int = 40,
                       verbose::Bool = true)

Train the network inside the UDE on observable data only (`observable_rollout_loss`) using
`Adam(adam_lr)` for `adam_iters` iterations, then `LBFGS()` for `lbfgs_iters` iterations.
Gradients come from `threaded_forward_gradient`; start Julia with `--threads=auto` to use them.

Returns `(; p_trained, loss_history, initial_loss, final_loss, holdout_metrics)`.
"""
function train_battery_ude!(
    model::HybridSPMeTModel,
    dataset::SyntheticTwinDataset;
    adam_lr::Real = 5.0e-3,
    adam_iters::Int = 100,
    lbfgs_iters::Int = 40,
    verbose::Bool = true,
)
    p0 = copy(model.p_init)
    loss_history = Float64[]

    obj_fn(p, _data = nothing) = observable_rollout_loss(p, model, dataset)
    loss_flat(q) = observable_rollout_loss(q, model, dataset)
    grad_fn!(G, p, _data = nothing) = (G .= threaded_forward_gradient(loss_flat, p); G)

    init_l = obj_fn(p0, nothing)
    push!(loss_history, Float64(init_l))
    if verbose
        println("  [Init]  Iteration   0 | Loss = $(round(init_l, sigdigits=6))")
    end

    function callback(state, l)
        push!(loss_history, Float64(l))
        iter = length(loss_history) - 1
        if verbose && (iter % 10 == 0 || iter == adam_iters || iter == adam_iters + lbfgs_iters)
            stage = iter <= adam_iters ? "Adam " : "LBFGS"
            println("  [$stage] Iteration $(lpad(iter, 3)) | Loss = $(round(l, sigdigits=6))")
        end
        return false
    end

    optf = Optimization.OptimizationFunction(obj_fn; grad = grad_fn!)

    res_adam = Optimization.solve(
        Optimization.OptimizationProblem(optf, p0),
        OptimizationOptimisers.Adam(Float64(adam_lr));
        maxiters = adam_iters,
        callback = callback,
    )
    p_curr = res_adam.u

    if lbfgs_iters > 0
        res_lbfgs = Optimization.solve(
            Optimization.OptimizationProblem(optf, p_curr),
            OptimizationOptimJL.LBFGS();
            maxiters = lbfgs_iters,
            callback = callback,
        )
        p_curr = res_lbfgs.u
    end

    final_l   = obj_fn(p_curr, nothing)
    p_trained = (p_curr isa ComponentVector) ? p_curr : ComponentVector(p_curr, model.p_axes)

    return (;
        p_trained,
        loss_history,
        initial_loss = Float64(init_l),
        final_loss   = Float64(final_l),
        holdout_metrics = evaluate_holdout_cycles(model, p_trained, dataset),
    )
end
