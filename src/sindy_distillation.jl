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
