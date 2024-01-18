using Test

include("../scripts/1D_maxwell_additive_source_lossy_layer.jl")

# ---- Unit test for 1D code update_E_z! ----
E_z = zeros(10)
H_y = ones(10)
E_z_e_loss = ones(10)
E_z_h_loss = ones(10)

# Call the update function
@parallel update_E_z!(H_y, E_z, E_z_e_loss, E_z_h_loss)

# Define the expected result manually based on the physics equation
expected_result_z = zeros(10)
for i in 2:10
    expected_result_z[i] = E_z_e_loss[i] * E_z[i] + E_z_h_loss[i] * (H_y[i] - H_y[i-1])
end

@testset "Unit Test: update_E_z" begin
    @test isapprox(E_z, expected_result_z)
end

# ---- Unit test for 1D code update_H_y! ----
H_y = zeros(10)
E_z = ones(10)
H_y_e_loss = ones(10)
H_y_h_loss = ones(10)

# Call the update function
@parallel update_H_y!(H_y, E_z, H_y_e_loss, H_y_h_loss)

# Define the expected result manually based on the physics equation
expected_result_y = zeros(10)
for i in 1:9
    expected_result_y[i] = H_y_h_loss[i] * H_y[i] + H_y_e_loss[i] * (E_z[i+1] - E_z[i])
end

@testset "Unit Test: update_H_y" begin
    @test isapprox(H_y, expected_result_y)
end

# ---- Reference test ----
nx_ = 200; nt_ = 450; nvis_ = 10


Ez = FDTD_1D(nx_, nt_, nvis_; src="exp", do_visu=false, do_test=false)

Ez_ref = zeros(Float32, nx_ + 1)
Ez_ref = load("../test/ref_Ez_1D_cpu.jld")
array_value = Ez_ref["data"]

@testset "Reference Test: Ez_ref ≈ Ez" begin
    @test isapprox(Ez, array_value)
end

