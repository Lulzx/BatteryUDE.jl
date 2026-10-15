# ============================================================================
# scripts/train_synthetic_twin.jl
# End-to-End Synthetic Digital Twin Training, Unseen Cycle Extrapolation,
# and Symbolic SINDy Law Recovery
# ============================================================================

using Random
using Printf
using Statistics
using BatteryUDE

function main()
    println("======================================================================")
    println("BatteryUDE.jl — Phase 4: End-to-End Synthetic Digital Twin & SINDy")
    println("======================================================================")

    rng = Random.MersenneTwister(2026)
    cell = chen2020_params()

    # 1. Generate 25-cycle synthetic ground-truth dataset (18 train, 7 holdout)
    println("\n[Step 1] Generating 25 accelerated cycles with hidden Tafel-diffusion SEI law...")
    dataset = generate_synthetic_twin_data(
        cell;
        N_cycles = 25,
        train_cycles = 18,
        t_cycle = 120.0,
        dt_save = 10.0,
        i0_sei_true = 2.0e-6,
        U_sei_ref = 0.40,
        accel_factor = 150.0,
        noise_std_V = 1.0e-3,
        rng = rng,
    )

    c1_end  = dataset.cycles[1]
    c18_end = dataset.cycles[18]
    c25_end = dataset.cycles[25]
    @printf("  Cycle  1 End: delta_sei_tilde = %.4f | Q_lli = %.5f Ah\n", c1_end.delta_sei_end, c1_end.Q_lli_end)
    @printf("  Cycle 18 End: delta_sei_tilde = %.4f | Q_lli = %.5f Ah\n", c18_end.delta_sei_end, c18_end.Q_lli_end)
    @printf("  Cycle 25 End: delta_sei_tilde = %.4f | Q_lli = %.5f Ah\n", c25_end.delta_sei_end, c25_end.Q_lli_end)

    # 2. Initialize HybridSPMeTModel with Lux.Chain SEI neural surrogate
    println("\n[Step 2] Initializing HybridSPMeTModel (Lux.Chain 4 -> 24 -> 24 -> 1)...")
    model = HybridSPMeTModel(
        cell;
        rng = rng,
        hidden_dim = 24,
        init_scale = 0.10,
        j_scale = 0.30,
        enable_thermal = true,
        enable_pore_clogging = true,
    )

    init_holdout = evaluate_holdout_cycles(model, model.p_init, dataset)
    @printf("  Pre-Training Holdout Voltage RMSE (Cycles 19-25): %.3f mV\n", init_holdout.voltage_rmse_mV)
    @printf("  Pre-Training Holdout Capacity RMSE (Cycles 19-25): %.5f Ah\n", init_holdout.capacity_rmse_Ah)

    # 3. Train using Adam(5e-3) for 100 iterations + L-BFGS for 40 iterations
    println("\n[Step 3] Training UDE via Adam(5e-3, 100 iters) + L-BFGS(40 iters)...")
    train_res = train_battery_ude!(
        model,
        dataset;
        adam_lr = 5.0e-3,
        adam_iters = 100,
        lbfgs_iters = 40,
        anchor_cycles = Int[],
        verbose = true,
    )

    loss_ratio = train_res.initial_loss / max(train_res.final_loss, 1.0e-16)
    println("\n[Step 4] Verification Metrics:")
    @printf("  Initial Training Loss : %.6e\n", train_res.initial_loss)
    @printf("  Final Training Loss   : %.6e\n", train_res.final_loss)
    @printf("  Loss Reduction Factor : %.2fx (%.2f orders of magnitude)\n", loss_ratio, log10(loss_ratio))
    @printf("  Holdout Voltage RMSE  : %.4f mV (Gate: < 5.0 mV)\n", train_res.holdout_metrics.voltage_rmse_mV)
    @printf("  Holdout Capacity RMSE : %.6f Ah\n", train_res.holdout_metrics.capacity_rmse_Ah)

    # 4. Symbolic SINDy Distillation (STLSQ)
    println("\n[Step 5] Distilling learned neural network into closed-form analytical law...")
    sindy_res = distill_sei_sindy(
        model,
        train_res.p_trained;
        eta_sei_grid = range(-0.34, -0.27, length = 15),
        delta_sei_grid = range(0.05, 1.50, length = 15),
        theta_n_grid = range(0.77, 0.80, length = 4),
        U_sei_ref = dataset.U_sei_ref,
        lambda_ridge = 1.0e-4,
        rel_threshold = 0.25,
        print_result = true,
    )

    @printf("  True Effective -i0_sei : %.4e A/m^2\n", -dataset.i0_sei_true)
    @printf("  SINDy Coefficient xi_3 : %.4e A/m^2\n", sindy_res.coefficients[3])

    @assert loss_ratio >= 100.0 "Expected training loss to decrease by at least 2 orders of magnitude (got $(loss_ratio)x)"
    @assert train_res.holdout_metrics.voltage_rmse_mV < 5.0 "Expected holdout voltage RMSE < 5.0 mV (got $(train_res.holdout_metrics.voltage_rmse_mV) mV)"

    println("\nSUCCESS: All Phase 4 Digital Twin & Symbolic Recovery gates passed!")
    return (train_res, sindy_res)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
