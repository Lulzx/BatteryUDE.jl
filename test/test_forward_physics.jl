# ============================================================================
# test/test_forward_physics.jl
# Phase 2 Verification Gate — Test 2: 1C Constant-Current Discharge Curve
# ============================================================================

using Test
using BatteryUDE
using OrdinaryDiffEq

@testset "Phase 2 - 1C Discharge Curve & Voltage Bounds" begin
    cell = chen2020_params()
    u0 = initial_state(
        cell;
        theta_n0 = 0.85,
        theta_p0 = 0.25,
        c_e0_tilde = 1.0,
        delta_sei0_tilde = 0.0,
        Q_lli0 = 0.0,
        T0_tilde = 0.0,
    )

    I_1C = cell.I_1C # 5.0 A constant discharge
    I_app_fn(t) = I_1C

    V0 = terminal_voltage(u0, I_app_fn(0.0), cell)
    @test 4.1 <= V0 <= 4.3

    cb = voltage_cutoff_callback(cell, I_app_fn; V_min = 2.8)
    f_discharge(u, p, t) = spmet_rhs(
        u,
        cell,
        t,
        I_app_fn;
        j_deg_fn = nothing,
        enable_thermal = true,
        enable_pore_clogging = true,
    )

    tspan = (0.0, 4500.0)
    prob  = ODEProblem{false}(f_discharge, u0, tspan)
    sol   = solve(
        prob,
        Rodas5P();
        callback = cb,
        reltol = 1.0e-7,
        abstol = 1.0e-9,
        saveat = 30.0,
        save_end = false,
    )

    V_curve = [terminal_voltage(u_k, I_app_fn(t_k), cell) for (u_k, t_k) in zip(sol.u, sol.t)]

    @info "1C Discharge completed" V_start=V_curve[1] V_end=V_curve[end] t_end=sol.t[end] num_points=length(V_curve)

    @test 4.1 <= V_curve[1] <= 4.3
    @test isapprox(V_curve[end], 2.8; atol = 2.0e-2)
    # Assert monotonic decrease across saved points
    @test all(diff(V_curve) .< 0.0)
end
