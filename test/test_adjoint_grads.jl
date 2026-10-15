# ============================================================================
# test/test_adjoint_grads.jl
# Phase 3 Verification Gate — Adjoint vs ForwardDiff Gradient Check
# ============================================================================

using Test
using LinearAlgebra
using Random
using BatteryUDE
using OrdinaryDiffEq
using SciMLSensitivity
using ForwardDiff
using Zygote

@testset "Phase 3 - Adjoint vs ForwardDiff Gradient Verification" begin
    rng  = Random.MersenneTwister(1234)
    cell = chen2020_params(N_r = 8, N_n = 4, N_s = 2, N_p = 4)

    model = HybridSPMeTModel(
        cell;
        rng = rng,
        hidden_dim = 24,
        init_scale = 0.25,
        j_scale = 0.05,
        enable_thermal = true,
        enable_pore_clogging = true,
    )

    u0 = initial_state(
        cell;
        theta_n0 = 0.75,
        theta_p0 = 0.35,
        c_e0_tilde = 1.0,
        delta_sei0_tilde = 0.2,
        Q_lli0 = 0.0,
        T0_tilde = 0.0,
    )

    # 600-second dynamic charge/discharge trajectory
    t_save = collect(0.0:60.0:600.0)
    I_app_fn(t) = 4.0 * cos(2.0 * pi * t / 600.0) - 1.5

    # Generate synthetic target trajectory with perturbed parameters p_target
    p_eval   = copy(model.p_init)
    p_target = copy(model.p_init) .+ 0.15

    prob_target = build_ude_problem(model, u0, (0.0, 600.0), I_app_fn; p = p_target)
    sol_target  = solve(
        prob_target,
        Rodas5P();
        saveat = t_save,
        reltol = 1.0e-10,
        abstol = 1.0e-12,
    )

    N_r = cell.N_r
    N_x = num_electrolyte_cells(cell)
    idx_Q_lli = 2 * N_r + N_x + 4

    V_target = [terminal_voltage(sol_target.u[k], I_app_fn(t_save[k]), cell) for k in eachindex(t_save)]
    Q_target = sol_target.u[end][idx_Q_lli]

    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())

    loss_fn(p) = adjoint_trajectory_loss(
        p,
        model,
        u0,
        t_save,
        I_app_fn,
        V_target,
        Q_target;
        w_V = 1.0,
        w_Q = 100.0,
        sensealg = sensealg,
        solver = Rodas5P(),
        reltol = 1.0e-10,
        abstol = 1.0e-12,
    )

    g_exact = ForwardDiff.gradient(loss_fn, p_eval)
    g_adj   = Zygote.gradient(loss_fn, p_eval)[1]

    cos_sim = dot(g_exact, g_adj) / (norm(g_exact) * norm(g_adj))
    rel_err = norm(g_exact .- g_adj) / norm(g_exact)

    @info "Adjoint gradient verification" cos_sim rel_err norm_exact=norm(g_exact) norm_adj=norm(g_adj)

    @test cos_sim > 0.999
    @test rel_err < 1.0e-4
end
