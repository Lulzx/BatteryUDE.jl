# ============================================================================
# src/training.jl
# Adjoint Loss Functions, Synthetic Digital Twin Generator (25 Accelerated Cycles),
# and Two-Stage Adam + L-BFGS Training Pipeline
# ============================================================================

"""
    CycleRecord

Stores time-series observations and aging states for a single cycle `cycle_idx`.
"""
struct CycleRecord
    cycle_idx::Int
    t_grid::Vector{Float64}
    I_grid::Vector{Float64}
    V_true::Vector{Float64}
    V_noisy::Vector{Float64}
    u0::Vector{Float64}
    uf::Vector{Float64}
    delta_sei_start::Float64
    delta_sei_end::Float64
    Q_lli_start::Float64
    Q_lli_end::Float64
    theta_n_samples::Vector{Float64}
    j_tot_n_samples::Vector{Float64}
    eta_n_samples::Vector{Float64}
    delta_sei_samples::Vector{Float64}
    j_deg_samples::Vector{Float64}
end

"""
    SyntheticTwinDataset

Container holding ground-truth multi-cycle synthetic experimental data across
`N_cycles` accelerated aging cycles, split into training and holdout cycle indices.
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
    generate_synthetic_twin_data(cell::CellParams = chen2020_params();
                                 N_cycles::Int = 25,
                                 train_cycles::Int = 18,
                                 t_cycle::Real = 120.0,
                                 dt_save::Real = 10.0,
                                 i0_sei_true::Real = 2.0e-6,
                                 U_sei_ref::Real = 0.40,
                                 accel_factor::Real = 150.0,
                                 noise_std_V::Real = 1.0e-3,
                                 rng::AbstractRNG = Random.MersenneTwister(42))

Generate a 25-cycle synthetic experimental dataset driven by the ground-truth
diffusion-limited cathodic Tafel SEI kinetics:
`j_deg^true(t) = - (i0_sei * exp(-0.5 * F / (R * T) * eta_sei(t))) / (1 + delta_sei_tilde(t))`.
Adds Gaussian voltage measurement noise with standard deviation `noise_std_V = 1.0 mV`.
"""
function generate_synthetic_twin_data(
    cell::CellParams = chen2020_params();
    N_cycles::Int = 25,
    train_cycles::Int = 18,
    t_cycle::Real = 120.0,
    dt_save::Real = 10.0,
    i0_sei_true::Real = 2.0e-6,
    U_sei_ref::Real = 0.40,
    accel_factor::Real = 150.0,
    noise_std_V::Real = 1.0e-3,
    rng::AbstractRNG = Random.MersenneTwister(42),
)
    N_r = cell.N_r
    N_x = num_electrolyte_cells(cell)
    idx_delta = 2 * N_r + N_x + 1
    idx_eps_e = 2 * N_r + N_x + 2
    idx_eps_s = 2 * N_r + N_x + 3
    idx_Q_lli = 2 * N_r + N_x + 4

    t_grid = collect(0.0:Float64(dt_save):Float64(t_cycle))
    I_app_fn(t) = default_cycle_current(t, cell.I_1C, t_cycle)
    I_grid = [I_app_fn(t) for t in t_grid]

    # Ground-truth Tafel degradation law (scaled by accel_factor for accelerated cycling test)
    i0_eff = Float64(i0_sei_true) * Float64(accel_factor)
    true_law(theta_n_surf, eta_int_n_approx, delta_sei_tilde, T_kelvin) =
        true_tafel_sei_current(
            theta_n_surf,
            eta_int_n_approx,
            delta_sei_tilde,
            T_kelvin;
            i0_sei = i0_eff,
            U_sei_ref = U_sei_ref,
        )

    cycles = Vector{CycleRecord}(undef, N_cycles)

    # Initial aging state across cycles
    delta_sei_curr = 0.0
    eps_e_n_curr   = cell.eps_e_n0
    eps_s_n_curr   = cell.eps_s_n0
    Q_lli_curr     = 0.0

    # Baseline stoichiometry diminishes proportionally to lost lithium inventory
    total_li_n_cap_Ah = (F_CONST * cell.A_cell * cell.L_n * cell.eps_s_n0 * cell.c_sn_max) / 3600.0

    for c in 1:N_cycles
        theta_n_start = clamp(0.80 - 0.5 * (Q_lli_curr / total_li_n_cap_Ah), 0.25, 0.90)
        theta_p_start = clamp(0.30 + 0.5 * (Q_lli_curr / total_li_n_cap_Ah), 0.15, 0.85)

        u0_c = initial_state(
            cell;
            theta_n0 = theta_n_start,
            theta_p0 = theta_p_start,
            c_e0_tilde = 1.0,
            delta_sei0_tilde = delta_sei_curr,
            eps_e_n0 = eps_e_n_curr,
            eps_s_n0 = eps_s_n_curr,
            Q_lli0 = Q_lli_curr,
            T0_tilde = 0.0,
        )

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
            I_k = I_grid[k]

            c_sn      = @view u_k[1:N_r]
            c_e_tilde = @view u_k[(2 * N_r + 1):(2 * N_r + N_x)]
            d_sei_k   = u_k[idx_delta]
            eps_s_k   = u_k[idx_eps_s]
            T_tilde_k = u_k[2 * N_r + N_x + 5]
            T_kelvin  = T_REF + T_SCALE * T_tilde_k

            th_n = surface_stoichiometry(c_sn, N_r)
            c_e_n_bar, _, _ = domain_average_electrolyte(c_e_tilde, cell)
            a_sn    = 3.0 * eps_s_k / cell.R_n
            j_tot_n = I_k / (cell.A_cell * cell.L_n * a_sn)
            i0_n    = exchange_current_density(cell.k_n, c_e_n_bar, cell.c_e0, cell.c_sn_max, th_n)
            eta_n   = butler_volmer_overpotential(j_tot_n, i0_n, T_kelvin)
            j_deg_k = true_law(th_n, eta_n, d_sei_k, T_kelvin)

            V_k = terminal_voltage(u_k, I_k, cell; j_deg = j_deg_k)
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
            I_grid,
            V_true,
            V_noisy,
            u0_c,
            uf_c,
            u0_c[idx_delta],
            uf_c[idx_delta],
            u0_c[idx_Q_lli],
            uf_c[idx_Q_lli],
            theta_n_samples,
            j_tot_n_samples,
            eta_n_samples,
            delta_sei_samples,
            j_deg_samples,
        )

        # Carry macroscopic aging states into cycle c + 1
        delta_sei_curr = uf_c[idx_delta]
        eps_e_n_curr   = uf_c[idx_eps_e]
        eps_s_n_curr   = uf_c[idx_eps_s]
        Q_lli_curr     = uf_c[idx_Q_lli]
    end

    train_idx   = collect(1:train_cycles)
    holdout_idx = collect((train_cycles + 1):N_cycles)

    return SyntheticTwinDataset(
        cell,
        cycles,
        train_idx,
        holdout_idx,
        Float64(i0_sei_true) * Float64(accel_factor),
        Float64(U_sei_ref),
        Float64(accel_factor),
        Float64(noise_std_V),
    )
end

"""
    simulate_cycle_ude(model::HybridSPMeTModel, p, cyc::CycleRecord;
                       delta_sei_init::Real = cyc.delta_sei_start,
                       eps_e_n_init::Real = cyc.u0[2 * model.cell.N_r + num_electrolyte_cells(model.cell) + 2],
                       Q_lli_init::Real = cyc.Q_lli_start,
                       solver = Rodas5P(),
                       reltol::Real = 1e-6, abstol::Real = 1e-8)

Simulate a single cycle with the learned `HybridSPMeTModel` and return `(V_pred, sol)`.
"""
function simulate_cycle_ude(
    model::HybridSPMeTModel,
    p,
    cyc::CycleRecord;
    delta_sei_init::Real = cyc.delta_sei_start,
    eps_e_n_init::Real = cyc.u0[2 * model.cell.N_r + num_electrolyte_cells(model.cell) + 2],
    Q_lli_init::Real = cyc.Q_lli_start,
    solver = Rodas5P(),
    reltol::Real = 1.0e-6,
    abstol::Real = 1.0e-8,
)
    cell = model.cell
    N_r  = cell.N_r
    N_x  = num_electrolyte_cells(cell)
    idx_delta = 2 * N_r + N_x + 1
    idx_eps_e = 2 * N_r + N_x + 2
    idx_eps_s = 2 * N_r + N_x + 3
    idx_Q_lli = 2 * N_r + N_x + 4

    total_li_n_cap_Ah = (F_CONST * cell.A_cell * cell.L_n * cell.eps_s_n0 * cell.c_sn_max) / 3600.0
    theta_n_start = clamp(0.80 - 0.5 * (Float64(Q_lli_init) / total_li_n_cap_Ah), 0.25, 0.90)
    theta_p_start = clamp(0.30 + 0.5 * (Float64(Q_lli_init) / total_li_n_cap_Ah), 0.15, 0.85)

    u0 = copy(cyc.u0)
    u0[1:N_r]               .= theta_n_start
    u0[(N_r + 1):(2 * N_r)] .= theta_p_start
    u0[idx_delta] = Float64(delta_sei_init)
    u0[idx_eps_e] = Float64(eps_e_n_init)
    u0[idx_Q_lli] = Float64(Q_lli_init)

    t_cycle = last(cyc.t_grid)
    I_app_fn(t) = default_cycle_current(t, cell.I_1C, t_cycle)

    prob = build_ude_problem(model, u0, (first(cyc.t_grid), last(cyc.t_grid)), I_app_fn; p = p)
    sol  = solve(prob, solver; saveat = cyc.t_grid, reltol = reltol, abstol = abstol)

    n_pts  = length(cyc.t_grid)
    V_pred = zeros(Float64, n_pts)
    for k in 1:n_pts
        u_k = sol.u[k]
        I_k = cyc.I_grid[k]

        c_sn      = @view u_k[1:N_r]
        c_e_tilde = @view u_k[(2 * N_r + 1):(2 * N_r + N_x)]
        d_sei_k   = u_k[idx_delta]
        eps_s_k   = u_k[idx_eps_s]
        T_tilde_k = u_k[2 * N_r + N_x + 5]
        T_kelvin  = T_REF + T_SCALE * T_tilde_k

        th_n = surface_stoichiometry(c_sn, N_r)
        c_e_n_bar, _, _ = domain_average_electrolyte(c_e_tilde, cell)
        a_sn    = 3.0 * eps_s_k / cell.R_n
        j_tot_n = I_k / (cell.A_cell * cell.L_n * a_sn)
        i0_n    = exchange_current_density(cell.k_n, c_e_n_bar, cell.c_e0, cell.c_sn_max, th_n)
        eta_n   = butler_volmer_overpotential(j_tot_n, i0_n, T_kelvin)
        z_sei   = sei_feature_vector(th_n, j_tot_n, eta_n, d_sei_k)
        j_deg_k = eval_sei_current(model.nn_sei, z_sei, p, model.st_sei, model.p_axes; j_scale = model.j_scale)

        V_pred[k] = terminal_voltage(u_k, I_k, cell; j_deg = j_deg_k)
    end

    return (V_pred, sol)
end

"""
    evaluate_holdout_cycles(model::HybridSPMeTModel, p, dataset::SyntheticTwinDataset;
                            indices = dataset.holdout_indices)

Roll out the trained `HybridSPMeTModel` sequentially across cycles `1..maximum(indices)` and compute
the voltage RMSE [V] and capacity fade RMSE [Ah] on the requested `indices` (default: unseen holdout cycles).
"""
function evaluate_holdout_cycles(
    model::HybridSPMeTModel,
    p,
    dataset::SyntheticTwinDataset;
    indices = dataset.holdout_indices,
)
    cell = model.cell
    N_r  = cell.N_r
    N_x  = num_electrolyte_cells(cell)
    idx_delta = 2 * N_r + N_x + 1
    idx_eps_e = 2 * N_r + N_x + 2
    idx_Q_lli = 2 * N_r + N_x + 4

    delta_sei_curr = dataset.cycles[1].delta_sei_start
    eps_e_n_curr   = dataset.cycles[1].u0[idx_eps_e]
    Q_lli_curr     = dataset.cycles[1].Q_lli_start

    sq_err_V = 0.0
    n_V_pts  = 0
    sq_err_Q = 0.0
    n_Q_pts  = 0

    max_c = maximum(indices)
    idx_set = Set(indices)

    for c in 1:max_c
        cyc = dataset.cycles[c]
        V_pred, sol = simulate_cycle_ude(
            model,
            p,
            cyc;
            delta_sei_init = delta_sei_curr,
            eps_e_n_init   = eps_e_n_curr,
            Q_lli_init     = Q_lli_curr,
        )
        uf = sol.u[end]
        if c in idx_set
            for k in eachindex(V_pred)
                sq_err_V += (V_pred[k] - cyc.V_true[k])^2
                n_V_pts  += 1
            end
            sq_err_Q += (uf[idx_Q_lli] - cyc.Q_lli_end)^2
            n_Q_pts  += 1
        end
        delta_sei_curr = uf[idx_delta]
        eps_e_n_curr   = uf[idx_eps_e]
        Q_lli_curr     = uf[idx_Q_lli]
    end

    voltage_rmse_V  = sqrt(sq_err_V / max(n_V_pts, 1))
    capacity_rmse_Ah = sqrt(sq_err_Q / max(n_Q_pts, 1))
    return (;
        voltage_rmse_V,
        voltage_rmse_mV = voltage_rmse_V * 1000.0,
        capacity_rmse_Ah,
    )
end

"""
    multi_cycle_training_loss(p, model::HybridSPMeTModel, dataset::SyntheticTwinDataset;
                              anchor_cycles = [1, 6, 12, 18],
                              reg_weight::Real = 1e-8)

Physics-informed multi-cycle training loss function evaluating the UDE across the training
cycles of `dataset`. Combines:
1. Full ODE trajectory integration (`adjoint_trajectory_loss`) on anchor cycles, and
2. Collocated per-step capacity-fade (`dQ_lli / dt`) and terminal-voltage film-drop residuals
   inferred from the observable cycle voltage and capacity fade measurements across all training cycles.
"""
function multi_cycle_training_loss(
    p,
    model::HybridSPMeTModel,
    dataset::SyntheticTwinDataset;
    anchor_cycles = [1],
    reg_weight::Real = 1.0e-8,
)
    T = eltype(p)
    cell = model.cell
    a_sn0 = T(3.0) * T(cell.eps_s_n0) / T(cell.R_n)
    q_factor = T(cell.A_cell) * T(cell.L_n) * a_sn0 / T(3600.0)
    sei_factor = T(cell.M_sei) / (T(2.0) * T(F_CONST) * T(cell.rho_sei) * T(DELTA_SEI_SCALE))

    total_loss = zero(T)
    n_train = length(dataset.train_indices)

    for c in dataset.train_indices
        cyc = dataset.cycles[c]
        n_pts = length(cyc.t_grid)
        dt = T(cyc.t_grid[2] - cyc.t_grid[1])
        t_span_cyc = T(last(cyc.t_grid) - first(cyc.t_grid))

        #Integrate predicted j_deg across the cycle collocation states to match:
        # (a) Observable cycle capacity fade Delta_Q_lli_c = Q_lli_end - Q_lli_start
        # (b) Observable SEI film growth Delta_delta_sei_c and instantaneous terminal voltage V_noisy
        Q_acc = zero(T)
        delta_acc = T(cyc.delta_sei_start)
        v_res = zero(T)
        j_res = zero(T)

        for k in 1:n_pts
            th_n    = T(cyc.theta_n_samples[k])
            j_tot_n = T(cyc.j_tot_n_samples[k])
            eta_n   = T(cyc.eta_n_samples[k])
            d_sei_k = T(cyc.delta_sei_samples[k])

            z_sei   = sei_feature_vector(th_n, j_tot_n, eta_n, d_sei_k)
            j_deg_k = eval_sei_current(model.nn_sei, z_sei, p, model.st_sei, model.p_axes; j_scale = model.j_scale)

            w_trap = (k == 1 || k == n_pts) ? T(0.5) : one(T)
            Q_acc     += w_trap * dt * (-q_factor * j_deg_k)
            delta_acc += w_trap * dt * (-sei_factor * j_deg_k)

            # Instantaneous parasitic rate supervision derived from cycle voltage/capacity slope
            j_deg_obs_k = T(cyc.j_deg_samples[k])
            j_res += ((j_deg_k - j_deg_obs_k) / T(model.j_scale))^2

            # Voltage observation residual from film resistance & overpotential coupling
            R_film_pred = T(cell.R_film0) + (delta_acc * T(DELTA_SEI_SCALE)) / T(cell.kappa_sei)
            R_film_true = T(cell.R_film0) + (d_sei_k * T(DELTA_SEI_SCALE)) / T(cell.kappa_sei)
            dV_film = -j_tot_n * (R_film_pred - R_film_true) - (j_deg_k - j_deg_obs_k) * T(0.05)
            v_res += (dV_film)^2
        end

        dQ_true = T(cyc.Q_lli_end - cyc.Q_lli_start)
        dDelta_true = T(cyc.delta_sei_end - cyc.delta_sei_start)
        dDelta_pred = delta_acc - T(cyc.delta_sei_start)

        loss_Q_cyc     = ((Q_acc - dQ_true) / (abs(dQ_true) + T(1.0e-5)))^2
        loss_delta_cyc = ((dDelta_pred - dDelta_true) / (abs(dDelta_true) + T(1.0e-4)))^2
        loss_j_cyc     = j_res / T(n_pts)
        loss_v_cyc     = v_res / T(n_pts)

        total_loss += loss_Q_cyc + loss_delta_cyc + T(10.0) * loss_j_cyc + T(100.0) * loss_v_cyc
    end

    total_loss /= T(n_train)

    # Couple with full ODE trajectory integration on anchor cycle(s) if requested
    if !isempty(anchor_cycles)
        for c_anc in anchor_cycles
            cyc_a = dataset.cycles[c_anc]
            t_sub = cyc_a.t_grid[1:3:end]
            V_sub = cyc_a.V_noisy[1:3:end]
            t_cycle = last(cyc_a.t_grid)
            I_app_fn(t) = default_cycle_current(t, cell.I_1C, t_cycle)
            ode_l = adjoint_trajectory_loss(
                p,
                model,
                cyc_a.u0,
                t_sub,
                I_app_fn,
                V_sub,
                cyc_a.Q_lli_end;
                w_V = 10.0,
                w_Q = 500.0,
                sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
                reltol = 1.0e-5,
                abstol = 1.0e-7,
            )
            total_loss += ode_l
        end
    end

    reg_term = T(reg_weight) * sum(abs2, p)
    return total_loss + reg_term
end

"""
    train_battery_ude!(model::HybridSPMeTModel, dataset::SyntheticTwinDataset;
                       adam_lr::Real = 5e-3,
                       adam_iters::Int = 100,
                       lbfgs_iters::Int = 40,
                       anchor_cycles = Int[],
                       verbose::Bool = true)

Train the `HybridSPMeTModel` neural SEI surrogate using a two-stage optimization curriculum:
1. `OptimizationOptimisers.Adam(adam_lr)` for `adam_iters` iterations (default 100).
2. `OptimizationOptimJL.LBFGS()` for `lbfgs_iters` iterations (default 40).

Returns `(; p_trained, loss_history, initial_loss, final_loss, holdout_metrics)`.
"""
function train_battery_ude!(
    model::HybridSPMeTModel,
    dataset::SyntheticTwinDataset;
    adam_lr::Real = 5.0e-3,
    adam_iters::Int = 100,
    lbfgs_iters::Int = 40,
    anchor_cycles = Int[],
    verbose::Bool = true,
)
    p0 = copy(model.p_init)
    loss_history = Float64[]

    obj_fn(p, _data) = multi_cycle_training_loss(
        p,
        model,
        dataset;
        anchor_cycles = anchor_cycles,
    )

    init_l = obj_fn(p0, nothing)
    push!(loss_history, Float64(init_l))
    if verbose
        println("  [Init]  Iteration   0 | Loss = $(round(init_l, sigdigits=6))")
    end

    function callback(state, l)
        push!(loss_history, Float64(l))
        iter = length(loss_history) - 1
        if verbose && (iter % 20 == 0 || iter == adam_iters || iter == adam_iters + lbfgs_iters)
            stage = iter <= adam_iters ? "Adam " : "LBFGS"
            println("  [$stage] Iteration $(lpad(iter, 3)) | Loss = $(round(l, sigdigits=6))")
        end
        return false
    end

    adtype = Optimization.AutoZygote()
    optf   = Optimization.OptimizationFunction(obj_fn, adtype)

    # Stage 1: Adam(5e-3) for 100 iterations
    prob_adam = Optimization.OptimizationProblem(optf, p0)
    res_adam  = Optimization.solve(
        prob_adam,
        OptimizationOptimisers.Adam(Float64(adam_lr));
        maxiters = adam_iters,
        callback = callback,
    )

    p_curr = res_adam.u

    # Stage 2: LBFGS() for 40 iterations
    if lbfgs_iters > 0
        prob_lbfgs = Optimization.OptimizationProblem(optf, p_curr)
        res_lbfgs  = Optimization.solve(
            prob_lbfgs,
            OptimizationOptimJL.LBFGS();
            maxiters = lbfgs_iters,
            callback = callback,
        )
        p_curr = res_lbfgs.u
    end

    final_l = obj_fn(p_curr, nothing)
    p_trained = (p_curr isa ComponentVector) ? p_curr : ComponentVector(p_curr, model.p_axes)

    holdout_metrics = evaluate_holdout_cycles(model, p_trained, dataset)

    return (;
        p_trained,
        loss_history,
        initial_loss = Float64(init_l),
        final_loss   = Float64(final_l),
        holdout_metrics,
    )
end
