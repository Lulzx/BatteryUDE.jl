module BatteryUDE

using LinearAlgebra
using Random
using Statistics
using OrdinaryDiffEq
using SciMLSensitivity
using Lux
using ComponentArrays
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Zygote
using ForwardDiff

include("parameters.jl")
include("fvm_discretization.jl")
include("electrochemistry.jl")
include("neural_surrogates.jl")
include("hybrid_ode.jl")
include("training.jl")
include("sindy_distillation.jl")

export F_CONST, R_GAS, T_REF, T_SCALE, DELTA_SEI_SCALE, J_DEG_SCALE
export CellParams, chen2020_params, num_electrolyte_cells, state_dimension
export solid_diffusivity, initial_state
export surface_stoichiometry, max_radial_concentration_gradient
export spherical_fvm_rhs, spherical_fvm_rhs!
export electrolyte_fvm_rhs, electrolyte_fvm_rhs!
export domain_average_electrolyte, solid_lithium_inventory, electrolyte_lithium_inventory
export ocp_graphite, ocp_nmc811, exchange_current_density, butler_volmer_overpotential
export electrolyte_overpotential, terminal_voltage, thermal_rhs
export ad_softplus, build_sei_network, build_lam_network, init_sei_surrogate
export sei_feature_vector, eval_sei_current, true_tafel_sei_current
export HybridSPMeTModel, ude_rhs, spmet_rhs, voltage_cutoff_callback, build_ude_problem
export CycleRecord, SyntheticTwinDataset, default_cycle_current
export adjoint_trajectory_loss, generate_synthetic_twin_data
export simulate_cycle_ude, evaluate_holdout_cycles, multi_cycle_training_loss, train_battery_ude!
export SINDyDistillationResult, stlsq_sparse_regression, distill_sei_sindy

end # module BatteryUDE
