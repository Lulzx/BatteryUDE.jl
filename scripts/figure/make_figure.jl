# ============================================================================
# scripts/figure/make_figure.jl
# Build the README results figure from the CSVs written by
# scripts/train_synthetic_twin.jl. Uses its own environment so the package
# does not depend on a plotting library.
#
# Usage:
#   julia --project=scripts/figure -e 'using Pkg; Pkg.instantiate()'
#   julia --project=scripts/figure scripts/figure/make_figure.jl [tag]
# ============================================================================

using CairoMakie

const RESULTS = joinpath(@__DIR__, "..", "..", "docs", "results")
const ASSETS  = joinpath(@__DIR__, "..", "..", "docs", "assets")

# Light chart surface and validated categorical slots (blue, orange)
const SURFACE   = colorant"#fcfcfb"
const INK       = colorant"#0b0b0b"
const INK_2     = colorant"#52514e"
const MUTED     = colorant"#a3a29c"
const GRID      = colorant"#e7e6e2"
const BLUE      = colorant"#2a78d6"
const ORANGE    = colorant"#eb6834"
const HOLDOUT_BG = colorant"#f1f0ec"

function read_csv(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    cols = [String[] for _ in header]
    for ln in lines[2:end]
        for (i, v) in enumerate(split(ln, ","))
            push!(cols[i], v)
        end
    end
    parse_col(c) = all(v -> tryparse(Float64, v) !== nothing, c) ? parse.(Float64, c) : c
    return Dict(String(h) => parse_col(c) for (h, c) in zip(header, cols))
end

function main(tag = "standard")
    loss  = read_csv(joinpath(RESULTS, "$(tag)_loss.csv"))
    cyc   = read_csv(joinpath(RESULTS, "$(tag)_cycles.csv"))
    v25   = read_csv(joinpath(RESULTS, "$(tag)_voltage_cycle25.csv"))
    par   = read_csv(joinpath(RESULTS, "$(tag)_parity.csv"))
    summ  = Dict(r[1] => r[2] for r in (split(l, ",") for l in readlines(joinpath(RESULTS, "$(tag)_summary.csv"))[2:end]))
    num(k) = parse(Float64, summ[k])

    set_theme!(Theme(
        fontsize = 15,
        backgroundcolor = SURFACE,
        textcolor = INK,
        Axis = (backgroundcolor = SURFACE, xgridcolor = GRID, ygridcolor = GRID,
                leftspinecolor = MUTED, bottomspinecolor = MUTED, topspinevisible = false, rightspinevisible = false,
                xtickcolor = MUTED, ytickcolor = MUTED, xticklabelcolor = INK_2, yticklabelcolor = INK_2,
                titlealign = :left, titlesize = 16, titlefont = :bold),
        Legend = (framevisible = false, backgroundcolor = (SURFACE, 0.0), labelcolor = INK_2),
    ))

    fig = Figure(size = (1400, 980), figure_padding = 24)

    # (a) Holdout voltage, cycle 25
    is_hold = cyc["split"] .== "holdout"
    c25_rate = cyc["c_rate"][25]
    v_rmse = 1e3 * sqrt(sum(abs2, v25["V_pred"] .- v25["V_true"]) / length(v25["V_true"]))
    ax1 = Axis(fig[1, 1]; title = "(a) Holdout cycle 25 ($(c25_rate)C, unseen rate): voltage RMSE $(round(v_rmse, digits = 2)) mV",
               xlabel = "Time in cycle [s]", ylabel = "Terminal voltage [V]")
    lines!(ax1, v25["t"], v25["V_pred_untrained"]; color = MUTED, linewidth = 2, linestyle = :dash, label = "Untrained UDE")
    lines!(ax1, v25["t"], v25["V_true"]; color = INK, linewidth = 2, label = "True voltage")
    lines!(ax1, v25["t"], v25["V_pred"]; color = BLUE, linewidth = 2, label = "Trained UDE")
    scatter!(ax1, v25["t"], v25["V_noisy"]; color = (INK_2, 0.0), strokecolor = INK_2, strokewidth = 1.5, markersize = 9, label = "Measured (1 mV noise)")
    axislegend(ax1; position = :lb)

    # (b) Capacity fade across cycles
    n = length(cyc["cycle"])
    first_hold = findfirst(is_hold)
    ax2 = Axis(fig[1, 2]; title = "(b) Capacity fade: train on cycles 1–$(first_hold - 1), predict $(first_hold)–$(n)",
               xlabel = "Cycle", ylabel = "Lost lithium inventory [Ah]")
    vspan!(ax2, first_hold - 0.5, n + 0.5; color = HOLDOUT_BG)
    text!(ax2, first_hold - 0.2, 0.0; text = "holdout: 0.8C / 1.8C\n(unseen rates)", color = INK_2, fontsize = 13, align = (:left, :bottom))
    lines!(ax2, cyc["cycle"], cyc["Q_pred_untrained"]; color = MUTED, linewidth = 2, linestyle = :dash, label = "Untrained UDE")
    lines!(ax2, cyc["cycle"], cyc["Q_true"]; color = INK, linewidth = 2, label = "True")
    lines!(ax2, cyc["cycle"], cyc["Q_pred"]; color = BLUE, linewidth = 2, label = "Trained UDE")
    scatter!(ax2, cyc["cycle"], cyc["Q_meas"]; color = (INK_2, 0.0), strokecolor = INK_2, strokewidth = 1.5, markersize = 9, label = "Measured (1 mAh noise)")
    axislegend(ax2; position = :lt)
    xlims!(ax2, 0.5, n + 0.5)

    # (c) SEI rate parity against the hidden truth (evaluation only)
    tr = par["split"] .== "train"
    jt = -1e3 .* par["j_true"]
    jp = -1e3 .* par["j_pred"]
    ax3 = Axis(fig[2, 1]; title = "(c) Learned vs. hidden SEI current: R² $(round(num("parity_r2_train"), digits = 3)) train, $(round(num("parity_r2_holdout"), digits = 3)) holdout",
               xlabel = "True −j_deg [mA/m²]", ylabel = "Network −j_deg [mA/m²]", aspect = DataAspect())
    lo, hi = extrema(vcat(jt, jp))
    lines!(ax3, [lo, hi], [lo, hi]; color = MUTED, linewidth = 1.5, linestyle = :dash)
    scatter!(ax3, jt[tr], jp[tr]; color = BLUE, markersize = 8, strokecolor = SURFACE, strokewidth = 1, label = "Training cycles")
    scatter!(ax3, jt[.!tr], jp[.!tr]; color = ORANGE, markersize = 8, strokecolor = SURFACE, strokewidth = 1, label = "Holdout cycles")
    axislegend(ax3; position = :lt)

    # (d) Training loss
    it = loss["iteration"]
    ax4 = Axis(fig[2, 2]; title = "(d) Training loss (observables only, noise-normalized)",
               xlabel = "Optimizer iteration", ylabel = "Loss", yscale = log10)
    adam_end = 100
    vlines!(ax4, [adam_end]; color = MUTED, linestyle = :dash, linewidth = 1.5)
    text!(ax4, adam_end + 2, maximum(loss["loss"]); text = "Adam → L-BFGS", color = INK_2, fontsize = 13, align = (:left, :top))
    lines!(ax4, it, loss["loss"]; color = BLUE, linewidth = 2)

    mkpath(ASSETS)
    out = joinpath(ASSETS, tag == "standard" ? "synthetic_twin_results.png" : "synthetic_twin_results_$(tag).png")
    save(out, fig; px_per_unit = 2)
    println("Saved ", normpath(out))
end

main(length(ARGS) >= 1 ? ARGS[1] : "standard")
