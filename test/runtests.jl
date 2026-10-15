using Test
using BatteryUDE

@testset "BatteryUDE.jl Comprehensive Test Suite" begin
    include("test_conservation.jl")
    include("test_forward_physics.jl")
    include("test_adjoint_grads.jl")
end
