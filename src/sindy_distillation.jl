# ============================================================================
# src/sindy_distillation.jl
# Sparse Symbolic Regression (STLSQ / SINDy) for Distilling Trained Neural
# SEI Surrogates into Closed-Form Analytical Kinetic Laws
# ============================================================================

"""
    SINDyDistillationResult

Stores the results of symbolic SINDy distillation of the trained neural SEI surrogate.
"""
struct SINDyDistillationResult
    basis_names::Vector{String}
    coefficients::Vector{Float64}
    active_mask::BitVector
    r2_score::Float64
    rmse::Float64
    equation_str::String
    alpha_tafel::Float64
end

"""
    stlsq_sparse_regression(Theta::AbstractMatrix{Float64}, y::AbstractVector{Float64};
                            lambda_ridge::Real = 1e-4,
                            rel_threshold::Real = 0.15,
                            max_iter::Int = 10)

Column-normalized Sequential Thresholded Least Squares (STLSQ) with ridge regularization.
Normalizes columns of `Theta` to unit L2 norm during thresholding so basis functions with
different magnitudes are pruned fairly by relative contribution `|xi_norm_j| / max(|xi_norm|) < rel_threshold`.
"""
function stlsq_sparse_regression(
    Theta::AbstractMatrix{Float64},
    y::AbstractVector{Float64};
    lambda_ridge::Real = 1.0e-4,
    rel_threshold::Real = 0.15,
    max_iter::Int = 10,
)
    n_samples, n_basis = size(Theta)
    col_norms = [max(norm(@view(Theta[:, j])), 1.0e-12) for j in 1:n_basis]
    Theta_norm = Theta ./ reshape(col_norms, 1, n_basis)

    active = trues(n_basis)
    xi_norm = zeros(Float64, n_basis)

    for _ in 1:max_iter
        idx = findall(active)
        isempty(idx) && break

        Th_sub = @view Theta_norm[:, idx]
        G = Th_sub' * Th_sub + Float64(lambda_ridge) * I(length(idx))
        xi_sub = G \ (Th_sub' * y)

        fill!(xi_norm, 0.0)
        xi_norm[idx] .= xi_sub

        max_abs = maximum(abs, xi_sub)
        if max_abs > 0.0
            for j in idx
                if abs(xi_norm[j]) < Float64(rel_threshold) * max_abs
                    active[j] = false
                end
            end
        end
    end

    # Final unregularized (or lightly regularized) refit on the surviving active support
    idx_final = findall(active)
    fill!(xi_norm, 0.0)
    if !isempty(idx_final)
        Th_sub = @view Theta_norm[:, idx_final]
        G = Th_sub' * Th_sub + 1.0e-8 * I(length(idx_final))
        xi_norm[idx_final] .= G \ (Th_sub' * y)
    end

    xi_phys = xi_norm ./ col_norms
    return (xi_phys, active)
end

"""
    distill_sei_sindy(model::HybridSPMeTModel, p;
                      eta_sei_grid = range(-0.34, -0.27, length = 15),
                      delta_sei_grid = range(0.05, 1.50, length = 15),
                      theta_n_grid = range(0.77, 0.80, length = 4),
                      U_sei_ref::Real = 0.40,
                      lambda_ridge::Real = 1e-4,
                      rel_threshold::Real = 0.25,
                      print_result::Bool = true)

Evaluate the trained neural network `model.nn_sei` across a grid of `(eta_sei, delta_sei_tilde, theta_n_surf)`
states and perform sparse ridge regression (STLSQ) over the candidate basis library:
`{ 1, exp(-alpha * eta_sei), exp(-alpha * eta_sei) / (1 + delta_sei_tilde), theta_n_surf }`,
where `alpha = 0.5 * F / (R * T_ref)`.
"""
function distill_sei_sindy(
    model::HybridSPMeTModel,
    p;
    eta_sei_grid = range(-0.34, -0.27, length = 15),
    delta_sei_grid = range(0.05, 1.50, length = 15),
    theta_n_grid = range(0.77, 0.80, length = 4),
    U_sei_ref::Real = 0.40,
    lambda_ridge::Real = 1.0e-4,
    rel_threshold::Real = 0.25,
    print_result::Bool = true,
)
    cell  = model.cell
    alpha = 0.5 * F_CONST / (R_GAS * cell.T_ref)

    basis_names = [
        "1",
        "exp(-alpha * eta_sei)",
        "exp(-alpha * eta_sei) / (1 + delta_sei_tilde)",
        "theta_n_surf",
    ]

    n_total = length(eta_sei_grid) * length(delta_sei_grid) * length(theta_n_grid)
    Theta   = zeros(Float64, n_total, 4)
    y_nn    = zeros(Float64, n_total)

    row = 0
    for th_n in theta_n_grid
        U_n = ocp_graphite(Float64(th_n))
        i0_n = exchange_current_density(cell.k_n, 1.0, cell.c_e0, cell.c_sn_max, Float64(th_n))
        for eta_sei in eta_sei_grid
            eta_int_n = Float64(eta_sei) + Float64(U_sei_ref) - U_n
            j_tot_n   = 2.0 * i0_n * sinh((F_CONST / (2.0 * R_GAS * cell.T_ref)) * eta_int_n)
            exp_term  = exp(-alpha * Float64(eta_sei))

            for d_sei in delta_sei_grid
                row += 1
                z_sei = sei_feature_vector(Float64(th_n), j_tot_n, eta_int_n, Float64(d_sei))
                j_pred = eval_sei_current(model.nn_sei, z_sei, p, model.st_sei, model.p_axes; j_scale = model.j_scale)

                Theta[row, 1] = 1.0
                Theta[row, 2] = exp_term
                Theta[row, 3] = exp_term / (1.0 + Float64(d_sei))
                Theta[row, 4] = Float64(th_n)
                y_nn[row]     = Float64(j_pred)
            end
        end
    end

    xi, active = stlsq_sparse_regression(
        Theta,
        y_nn;
        lambda_ridge = lambda_ridge,
        rel_threshold = rel_threshold,
    )

    y_hat = Theta * xi
    ss_res = sum(abs2, y_nn .- y_hat)
    ss_tot = sum(abs2, y_nn .- mean(y_nn))
    r2     = 1.0 - ss_res / max(ss_tot, 1.0e-16)
    rmse   = sqrt(ss_res / n_total)

    terms = String[]
    for j in eachindex(xi)
        if active[j] && abs(xi[j]) > 0.0
            coeff_str = string(round(xi[j], sigdigits = 4))
            if basis_names[j] == "1"
                push!(terms, coeff_str)
            else
                push!(terms, "$(coeff_str) * $(basis_names[j])")
            end
        end
    end
    eq_rhs = isempty(terms) ? "0.0" : join(terms, " + ")
    eq_str = "j_deg(t) = " * eq_rhs

    if print_result
        println("======================================================================")
        println("Symbolic SINDy Distillation (STLSQ) Result:")
        println("  Recovered Law : $(eq_str)")
        println("  Tafel alpha   : $(round(alpha, digits=4)) V^-1 (0.5 * F / (R * T))")
        println("  R^2 Score     : $(round(r2, digits=5))")
        println("  Grid RMSE     : $(round(rmse, sigdigits=4)) A/m^2")
        println("======================================================================")
    end

    return SINDyDistillationResult(
        basis_names,
        xi,
        active,
        r2,
        rmse,
        eq_str,
        alpha,
    )
end

"""
    SINDyFreeformResult

Results of the free-form (log-space) SINDy distillation, where the Tafel slope `alpha`
and the SEI diffusion exponent are fitted rather than fixed.
"""
struct SINDyFreeformResult
    basis_names::Vector{String}
    coefficients::Vector{Float64}
    active_mask::BitVector
    r2_log::Float64
    r2_linear::Float64
    alpha_fit::Float64
    alpha_theory::Float64
    diffusion_exponent::Float64
    i0_fit::Float64
    equation_str::String
end

"""
    distill_sei_sindy_freeform(model::HybridSPMeTModel, p;
                               eta_sei_grid = range(-0.34, -0.27, length = 15),
                               delta_sei_grid = range(0.05, 1.50, length = 15),
                               theta_n_grid = range(0.77, 0.80, length = 4),
                               U_sei_ref::Real = 0.40,
                               lambda_ridge::Real = 1e-4,
                               rel_threshold::Real = 0.05,
                               print_result::Bool = true,
                               j_target_fn = nothing,
                               samples = nothing,
                               beta_ref::Real = 0.5)

Free-form SINDy distillation that does not assume the Tafel slope. The sign-constrained
network guarantees `j_deg < 0`, so the regression target is `log(-j_deg)` and STLSQ runs over
the generic library `{ 1, eta, eta^2, delta, log(1 + delta), theta_n, eta * delta }`
(with `eta` and `theta_n` centered on the sample means for conditioning). A Tafel-diffusion law
`j = -i0 * exp(-alpha * eta) / (1 + delta)^m` appears as the active set `{1, eta, log(1 + delta)}`
with `alpha = -xi_eta` and `m = -xi_log(1+delta)`. `alpha_theory = beta_ref * F / (R * T_ref)` is
reported for comparison only and does not enter the fit.

Pass `samples = (; theta_n, j_tot_n, eta_int_n, delta)` (vectors) to fit at specific states, such as
the states the model visits in a rollout, instead of the tensor grid.

Pass `j_target_fn(theta_n, eta_int_n, delta)` to fit a known function (e.g. the ground-truth law)
instead of the network, which checks the regression itself independently of training.
"""
function distill_sei_sindy_freeform(
    model::HybridSPMeTModel,
    p;
    eta_sei_grid = range(-0.34, -0.27, length = 15),
    delta_sei_grid = range(0.05, 1.50, length = 15),
    theta_n_grid = range(0.77, 0.80, length = 4),
    U_sei_ref::Real = 0.40,
    lambda_ridge::Real = 1.0e-4,
    rel_threshold::Real = 0.05,
    print_result::Bool = true,
    j_target_fn = nothing,
    samples = nothing,
    beta_ref::Real = 0.5,
)
    cell = model.cell
    alpha_theory = Float64(beta_ref) * F_CONST / (R_GAS * cell.T_ref)

    basis_names = [
        "1",
        "eta",
        "eta^2",
        "delta",
        "log(1 + delta)",
        "theta_n",
        "eta * delta",
    ]
    n_basis = length(basis_names)

    # Evaluation points (theta_n, j_tot_n, eta_int_n, delta): either supplied states or a tensor grid
    points = if samples === nothing
        pts = NTuple{4, Float64}[]
        for th_n in theta_n_grid
            U_n = ocp_graphite(Float64(th_n))
            i0_n = exchange_current_density(cell.k_n, 1.0, cell.c_e0, cell.c_sn_max, Float64(th_n))
            for eta_sei in eta_sei_grid
                eta_int_n = Float64(eta_sei) + Float64(U_sei_ref) - U_n
                j_tot_n   = 2.0 * i0_n * sinh((F_CONST / (2.0 * R_GAS * cell.T_ref)) * eta_int_n)
                for d_sei in delta_sei_grid
                    push!(pts, (Float64(th_n), j_tot_n, eta_int_n, Float64(d_sei)))
                end
            end
        end
        pts
    else
        collect(zip(Float64.(samples.theta_n), Float64.(samples.j_tot_n), Float64.(samples.eta_int_n), Float64.(samples.delta)))
    end

    eta_sei_of(pt) = ocp_graphite(pt[1]) + pt[3] - Float64(U_sei_ref)
    eta_bar = mean(eta_sei_of, points)
    th_bar  = mean(pt -> pt[1], points)

    n_total = length(points)
    Theta   = zeros(Float64, n_total, n_basis)
    j_nn    = zeros(Float64, n_total)

    for (row, pt) in enumerate(points)
        th_n, j_tot_n, eta_int_n, d = pt
        z_sei = sei_feature_vector(th_n, j_tot_n, eta_int_n, d)
        j_nn[row] = j_target_fn === nothing ?
            Float64(eval_sei_current(model.nn_sei, z_sei, p, model.st_sei, model.p_axes; j_scale = model.j_scale)) :
            Float64(j_target_fn(th_n, eta_int_n, d))

        x_eta = eta_sei_of(pt) - eta_bar
        Theta[row, 1] = 1.0
        Theta[row, 2] = x_eta
        Theta[row, 3] = x_eta^2
        Theta[row, 4] = d
        Theta[row, 5] = log1p(d)
        Theta[row, 6] = th_n - th_bar
        Theta[row, 7] = x_eta * d
    end

    all(<(0.0), j_nn) || error("Free-form SINDy needs a strictly cathodic network output (j_deg < 0).")
    y_log = log.(-j_nn)

    xi, active = stlsq_sparse_regression(
        Theta,
        y_log;
        lambda_ridge = lambda_ridge,
        rel_threshold = rel_threshold,
    )

    y_log_hat = Theta * xi
    r2_log    = 1.0 - sum(abs2, y_log .- y_log_hat) / max(sum(abs2, y_log .- mean(y_log)), 1.0e-16)
    j_hat     = -exp.(y_log_hat)
    r2_linear = 1.0 - sum(abs2, j_nn .- j_hat) / max(sum(abs2, j_nn .- mean(j_nn)), 1.0e-16)

    alpha_fit = -xi[2]
    m_fit     = -xi[5]
    # Undo the eta centering: log(i0) = xi_1 - xi_eta * eta_bar (meaningful when only {1, eta, log(1+delta)} survive)
    i0_fit    = exp(xi[1] - xi[2] * eta_bar)

    terms = String[]
    for j in eachindex(xi)
        if active[j] && abs(xi[j]) > 0.0
            coeff_str = string(round(xi[j], sigdigits = 4))
            push!(terms, basis_names[j] == "1" ? coeff_str : "$(coeff_str) * $(basis_names[j])")
        end
    end
    eq_str = "log(-j_deg) = " * (isempty(terms) ? "0.0" : join(terms, " + ")) *
             "   (eta, theta_n centered at $(round(eta_bar, sigdigits = 4)), $(round(th_bar, sigdigits = 4)))"

    if print_result
        println("======================================================================")
        println("Free-Form SINDy Distillation (log-space STLSQ, alpha not fixed):")
        println("  Library       : ", join(basis_names, ", "))
        println("  Active terms  : ", join(basis_names[active], ", "))
        println("  Recovered Law : $(eq_str)")
        println("  Fitted alpha  : $(round(alpha_fit, digits = 4)) V^-1 (reference beta F/(R T) = $(round(alpha_theory, digits = 4)))")
        println("  Diffusion exp : $(round(m_fit, digits = 4)) (theory 1)")
        println("  Fitted i0     : $(round(i0_fit, sigdigits = 4)) A/m^2")
        println("  R^2 (log)     : $(round(r2_log, digits = 5)) | R^2 (linear j_deg): $(round(r2_linear, digits = 5))")
        println("======================================================================")
    end

    return SINDyFreeformResult(
        basis_names,
        xi,
        active,
        r2_log,
        r2_linear,
        alpha_fit,
        alpha_theory,
        m_fit,
        i0_fit,
        eq_str,
    )
end
