using Test

include("../scripts/2D_maxwell_pml_xPU.jl")

# ---- Unit test Ex ----
Ex = zeros(5, 6)
Hz = ones(5, 6)
σ = 0.1
ε0 = 1.0
dt = 0.01
dy = 1.0

# Call the update function
@parallel update_Ex!(Ex, Hz, σ, ε0, dt, dy)

# Define the expected result manually based on the physics equation
expected_result_x = zeros(5, 6)
for i in 2:4
    for j in 2:5
        expected_result_x[i, j] = Ex[i, j] + dt / ε0 * (-σ * Ex[i, j] + (Hz[i, j+1] - Hz[i, j]) / dy)
    end
end
@testset "Unit Test: update_Ex" begin
    @test isapprox(Ex, expected_result_x)
end

# ---- Unit test Ey ----
Ey = zeros(6, 5)
Hz = ones(6, 5)
σ = 0.1
ε0 = 1.0
dt = 0.01
dx = 1.0

# Call the update function
@parallel update_Ey!(Ey, Hz, σ, ε0, dt, dy)

# Define the expected result manually based on the physics equation
expected_result_y = zeros(6, 5)
for i in 2:5
    for j in 2:4
        expected_result_y[i, j] = Ey[i, j] + dt / ε0 * (-σ * Ey[i, j] - (Hz[i, j] - Hz[i+1, j]) / dx)
    end
end
@testset "Unit Test: update_Ey" begin
    @test isapprox(Ey, expected_result_y)
end

# ---- Reference test ----
ny_ = 50; nt_ = 10; nvis_ = 10
nx_ = ny_ - 1 
pml_width = 10

Hz = maxwell(ny_, nt_, nvis_, pml_width; do_visu=false, do_check=true, do_test=true)

nx_pml, ny_pml = nx_ + 2 * pml_width, ny_ + 2 * pml_width

Hz_ref = zeros(Float32, nx_pml, ny_pml)
Hz_ref = load("../test/ref_Hz_2D_cpu.jld")
array_value = Hz_ref["data"]

@testset "Reference Test: Hz_ref = Hz" begin
    @test isapprox(Hz, array_value)
end