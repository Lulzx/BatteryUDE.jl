# ============================================================================
# test/test_conservation.jl
# Phase 2 Verification Gate — Test 1: Exact Lithium Mass Conservation (< 1e-12)
# ============================================================================

using Test
using BatteryUDE
using OrdinaryDiffEq

@testset "Phase 2 - Exact Lithium Mass Conservation" begin
    cell = chen2020_params()
    @test state_dimension(cell) == 45

    u0 = initial_state(
        cell;
        theta_n0 = 0.80,
        theta_p0 = 0.30,
        c_e0_tilde = 1.0,
        delta_sei0_tilde = 0.0,
        Q_lli0 = 0.0,
        T0_tilde = 0.0,
    )

    # Dynamic sinusoidal current profile I_app(t) = 5.0 * sin(2*pi*t / 600) for 1200 seconds
    I_app_fn(t) = 5.0 * sin(2.0 * pi * t / 600.0)

    # Pure SPMeT with j_deg = 0
    f_conserve(u, p, t) = spmet_rhs(
        u,
        cell,
        t,
        I_app_fn;
        j_deg_fn = nothing,
        enable_thermal = true,
        enable_pore_clogging = true,
    )

    tspan = (0.0, 1200.0)
    prob  = ODEProblem{false}(f_conserve, u0, tspan)
    sol   = solve(
        prob,
        Rodas5P();
        reltol = 1.0e-10,
        abstol = 1.0e-12,
        saveat = 0.0:30.0:1200.0,
    )

    @test sol.retcode == ReturnCode.Success

    N_solid_0 = solid_lithium_inventory(sol.u[1], cell)
    N_elec_0  = electrolyte_lithium_inventory(sol.u[1], cell)

    max_rel_err_solid = 0.0
    max_rel_err_elec  = 0.0

    for u_k in sol.u
        N_solid_k = solid_lithium_inventory(u_k, cell)
        N_elec_k  = electrolyte_lithium_inventory(u_k, cell)

        rel_err_s = abs(N_solid_k - N_solid_0) / N_solid_0
        rel_err_e = abs(N_elec_k - N_elec_0) / N_elec_0

        max_rel_err_solid = max(max_rel_err_solid, rel_err_s)
        max_rel_err_elec  = max(max_rel_err_elec, rel_err_e)
    end

    @info "Conservation check completed" max_rel_err_solid max_rel_err_elec
    @test max_rel_err_solid < 1.0e-12
    @test max_rel_err_elec  < 1.0e-12
end
