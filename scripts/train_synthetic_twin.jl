# ============================================================================
# scripts/train_synthetic_twin.jl
# Synthetic digital twin: generate variable C-rate aging data from a hidden SEI
# law, train the UDE on observable data only (noisy voltage + noisy capacity
# fade), evaluate on holdout cycles at unseen C-rates, and distill the network
# with both the fixed-alpha and the free-form SINDy libraries.
#
# Writes CSV results to docs/results/ (read by scripts/make_figure.jl).
#
# Usage: julia --project=. --threads=auto scripts/train_synthetic_twin.jl [beta_sei m_diff tag]
#   defaults: 0.5 1.0 standard   (the standard symmetric Tafel-diffusion law)
#   e.g.      0.35 2.0 blind     (a law outside the fixed-alpha library)
# ============================================================================

using Random
using Printf
using Statistics
using BatteryUDE

const RESULTS_DIR = joinpath(@__DIR__, "..", "docs", "results")

function write_csv(path, header, cols...)
    open(path, "w") do io
        println(io, join(header, ","))
        for i in eachindex(cols[1])
            println(io, join((string(c[i]) for c in cols), ","))
        end
    end
end

"""States the trained model itself visits when rolled out over the given cycles (no ground truth)."""
function rollout_samples(roll, dataset, cycles)
    th = Float64[]; jt = Float64[]; eta = Float64[]; d = Float64[]
    for c in cycles, k in eachindex(dataset.cycles[c].t_grid)
        th_n, j_tot_n, eta_n, d_sei, _ = sei_inputs(roll[c].states[k], dataset.cycles[c].I_grid[k], dataset.cell)
        push!(th, th_n); push!(jt, j_tot_n); push!(eta, eta_n); push!(d, d_sei)
    end
    return (; theta_n = th, j_tot_n = jt, eta_int_n = eta, delta = d)
end

function main(; beta_sei = 0.5, m_diff = 1.0, tag = "standard")
    println("======================================================================")
    println("BatteryUDE.jl — synthetic twin ($tag: beta_sei = $beta_sei, m_diff = $m_diff)")
    println("======================================================================")
    mkpath(RESULTS_DIR)

    rng  = Random.MersenneTwister(2026)
    cell = chen2020_params()

    println("\n[Step 1] Generating 25 accelerated cycles (C-rates vary per cycle)...")
    dataset = generate_synthetic_twin_data(cell; N_cycles = 25, train_cycles = 18, beta_sei = beta_sei, m_diff = m_diff, rng = rng)
    for c in (1, 18, 25)
        cy = dataset.cycles[c]
        @printf("  Cycle %2d: %.1fC | delta_sei_tilde = %.4f | Q_lli = %.5f Ah\n", c, cy.I_amp / cell.I_1C, cy.delta_sei_end, cy.Q_lli_end)
    end

    println("\n[Step 2] Initializing HybridSPMeTModel (Lux.Chain 4 -> 8 -> 8 -> 1, 121 parameters)...")
    model = HybridSPMeTModel(cell; rng = rng, hidden_dim = 8, init_scale = 0.10, j_scale = 0.30)
    init_holdout = evaluate_holdout_cycles(model, model.p_init, dataset)
    @printf("  Untrained holdout voltage RMSE : %.3f mV\n", init_holdout.voltage_rmse_mV)
    @printf("  Untrained holdout capacity RMSE: %.5f Ah\n", init_holdout.capacity_rmse_Ah)

    println("\n[Step 3] Training on observables only (noisy voltage + noisy capacity fade)...")
    t_train = @elapsed train_res = train_battery_ude!(model, dataset; adam_lr = 5.0e-3, adam_iters = 100, lbfgs_iters = 40)
    p = train_res.p_trained

    hold = train_res.holdout_metrics
    loss_ratio = train_res.initial_loss / train_res.final_loss
    println("\n[Step 4] Results:")
    @printf("  Training time          : %.0f s\n", t_train)
    @printf("  Loss (noise-normalized): %.4g -> %.4g (%.1fx)\n", train_res.initial_loss, train_res.final_loss, loss_ratio)
    @printf("  Holdout voltage RMSE   : %.3f mV (untrained %.2f mV; noise %.1f mV)\n", hold.voltage_rmse_mV, init_holdout.voltage_rmse_mV, 1e3 * dataset.noise_std_V)
    @printf("  Holdout capacity RMSE  : %.5f Ah\n", hold.capacity_rmse_Ah)
    for rate in sort(unique(dataset.cycles[c].I_amp for c in dataset.holdout_indices))
        idx = [c for c in dataset.holdout_indices if dataset.cycles[c].I_amp == rate]
        m = evaluate_holdout_cycles(model, p, dataset; indices = idx)
        @printf("    unseen %.1fC cycles %-12s: V RMSE %.3f mV | Q RMSE %.5f Ah\n", rate / cell.I_1C, string(idx), m.voltage_rmse_mV, m.capacity_rmse_Ah)
    end

    println("\n[Step 5] SEI rate vs. hidden truth (evaluation only, ground truth never used in training):")
    par_train = sei_rate_parity(model, p, dataset; indices = dataset.train_indices)
    par_hold  = sei_rate_parity(model, p, dataset; indices = dataset.holdout_indices)
    @printf("  Training cycles: R^2 = %.4f | relative RMSE = %.2f%%\n", par_train.r2, 100 * par_train.rel_rmse)
    @printf("  Holdout cycles : R^2 = %.4f | relative RMSE = %.2f%%\n", par_hold.r2, 100 * par_hold.rel_rmse)

    println("\n[Step 6] SINDy distillation")
    roll = rollout_cycles(model, p, dataset)
    samples = rollout_samples(roll, dataset, dataset.train_indices)

    println("\n  (a) Fixed-alpha library (alpha = 0.5 F/RT is given, only coefficients are fitted):")
    fixed = distill_sei_sindy(model, p; U_sei_ref = dataset.U_sei_ref, rel_threshold = 0.25, print_result = true)

    println("\n  (b) Free-form library on the states the model visits (alpha and exponent fitted):")
    free = distill_sei_sindy_freeform(model, p; samples = samples, U_sei_ref = dataset.U_sei_ref, beta_ref = beta_sei, print_result = true)

    println("\n  (c) Control: the same free-form fit applied to the hidden true law at the same states:")
    true_law(th, eta_int, d) = true_tafel_sei_current(th, eta_int, d, T_REF; i0_sei = dataset.i0_sei_true,
                                                      U_sei_ref = dataset.U_sei_ref, beta_sei = beta_sei, m_diff = m_diff)
    ctrl = distill_sei_sindy_freeform(model, p; samples = samples, U_sei_ref = dataset.U_sei_ref, beta_ref = beta_sei,
                                      j_target_fn = true_law, print_result = true)

    alpha_true = beta_sei * F_CONST / (R_GAS * cell.T_ref)
    println("\n  Summary (truth: alpha = $(round(alpha_true, digits = 3)) V^-1, m = $m_diff, i0 = $(dataset.i0_sei_true) A/m^2)")
    @printf("    %-24s alpha = %8.3f  m = %6.3f  i0 = %.3e  active = %s\n", "network, free-form:", free.alpha_fit, free.diffusion_exponent, free.i0_fit, join(free.basis_names[free.active_mask], ", "))
    @printf("    %-24s alpha = %8.3f  m = %6.3f  i0 = %.3e  active = %s\n", "true law, free-form:", ctrl.alpha_fit, ctrl.diffusion_exponent, ctrl.i0_fit, join(ctrl.basis_names[ctrl.active_mask], ", "))

    # ---------------------------------------------------------------- CSV output
    write_csv(joinpath(RESULTS_DIR, "$(tag)_loss.csv"), ["iteration", "loss"], 0:(length(train_res.loss_history) - 1), train_res.loss_history)

    roll0 = rollout_cycles(model, model.p_init, dataset)
    cyc_ids = eachindex(dataset.cycles)
    write_csv(joinpath(RESULTS_DIR, "$(tag)_cycles.csv"),
        ["cycle", "c_rate", "split", "Q_true", "Q_meas", "Q_pred", "Q_pred_untrained", "V_rmse_mV"],
        cyc_ids,
        [dataset.cycles[c].I_amp / cell.I_1C for c in cyc_ids],
        [c in dataset.train_indices ? "train" : "holdout" for c in cyc_ids],
        [dataset.cycles[c].Q_lli_end for c in cyc_ids],
        [dataset.cycles[c].Q_lli_meas_end for c in cyc_ids],
        [roll[c].Q_lli_end for c in cyc_ids],
        [roll0[c].Q_lli_end for c in cyc_ids],
        [1e3 * sqrt(mean(abs2, roll[c].V_pred .- dataset.cycles[c].V_true)) for c in cyc_ids])

    for c in (1, 25)
        cy = dataset.cycles[c]
        write_csv(joinpath(RESULTS_DIR, "$(tag)_voltage_cycle$(c).csv"),
            ["t", "I", "V_noisy", "V_true", "V_pred", "V_pred_untrained"],
            cy.t_grid, cy.I_grid, cy.V_noisy, cy.V_true, roll[c].V_pred, roll0[c].V_pred)
    end

    write_csv(joinpath(RESULTS_DIR, "$(tag)_parity.csv"), ["split", "j_true", "j_pred"],
        vcat(fill("train", length(par_train.j_true)), fill("holdout", length(par_hold.j_true))),
        vcat(par_train.j_true, par_hold.j_true), vcat(par_train.j_pred, par_hold.j_pred))

    open(joinpath(RESULTS_DIR, "$(tag)_summary.csv"), "w") do io
        println(io, "metric,value")
        for (k, v) in [
            "beta_sei" => beta_sei, "m_diff" => m_diff, "alpha_true" => alpha_true, "i0_true" => dataset.i0_sei_true,
            "train_seconds" => t_train, "loss_initial" => train_res.initial_loss, "loss_final" => train_res.final_loss,
            "holdout_V_rmse_mV" => hold.voltage_rmse_mV, "holdout_V_rmse_mV_untrained" => init_holdout.voltage_rmse_mV,
            "holdout_Q_rmse_Ah" => hold.capacity_rmse_Ah, "parity_r2_train" => par_train.r2, "parity_r2_holdout" => par_hold.r2,
            "fixed_xi3" => fixed.coefficients[3], "fixed_r2" => fixed.r2_score,
            "free_alpha" => free.alpha_fit, "free_m" => free.diffusion_exponent, "free_i0" => free.i0_fit,
            "free_r2_linear" => free.r2_linear, "free_active" => join(free.basis_names[free.active_mask], " + "),
            "ctrl_alpha" => ctrl.alpha_fit, "ctrl_m" => ctrl.diffusion_exponent, "ctrl_i0" => ctrl.i0_fit,
        ]
            println(io, k, ",", v)
        end
    end
    println("\nWrote results to $(normpath(RESULTS_DIR))/$(tag)_*.csv")

    return (; dataset, model, train_res, par_train, par_hold, fixed, free, ctrl)
end

if abspath(PROGRAM_FILE) == @__FILE__
    beta = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5
    m    = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
    tag  = length(ARGS) >= 3 ? ARGS[3] : "standard"
    main(; beta_sei = beta, m_diff = m, tag = tag)
end
