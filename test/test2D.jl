using Test

include("../scripts/2D_maxwell_pml_xPU.jl")

ny_ = 50; nt_ = 10; nvis_ = 10
nx_ = ny_ - 1 
pml_width = 10


Hz = maxwell(ny_, nt_, nvis_; do_visu=false, do_check=true, do_test=true)

nx_pml, ny_pml = nx_ + 2 * pml_width, ny_ + 2 * pml_width

# @testset "Unit Test: BC" begin
#     @test all(T[1,:] .== T[2,:])
#     @test all(T[end,:] .== T[end-1,:])
# end

Hz_ref = zeros(Float32, nx_pml, ny_pml)
Hz_ref = load("../test/ref_Hz_2D_cpu.jld")
array_value = Hz_ref["data"]

@testset "Reference Test: Hz_ref = Hz" begin
    @test array_value ≈ Hz
end