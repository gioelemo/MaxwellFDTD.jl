using Test

include("../scripts/3D_maxwell_pml_xPU.jl")

# ---- Unit test Ex ----
Ex = zeros(5, 6, 6)
Hy = ones(5, 4, 5)
Hz = ones(5, 5, 4)
σ = 0.1
ε0 = 1.0
dt = 0.01
dy = 1.0
dz = 1.0

# Call the update function
@parallel (1:size(Ex, 1), 1:size(Ex, 2) - 2, 1:size(Ex, 3) - 2) update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)

# Define the expected result manually based on the physics equation
expected_result_x = zeros(5, 6, 6)
for i in 1:5
    for j in 1:4
        for k in 1:4
            expected_result_x[i, j + 1, k + 1] = Ex[i, j + 1, k + 1] + dt / ε0 * (-σ * Ex[i, j + 1, k + 1] + (Hz[i, j + 1, k] - Hz[i, j, k]) / dy - (Hy[i, j, k + 1] - Hz[i, j, k]) / dz)
        end
    end
end
@testset "Unit Test: update_Ex" begin
    @test isapprox(Ex, expected_result_x)
end

# # ---- Unit test Ey ----
Ey = zeros(6, 5, 6)
Hz = ones(5, 5, 4)
Hx = ones(4, 5, 5)
σ = 0.1
ε0 = 1.0
dt = 0.01
dx = 1.0
dz = 1.0

# Call the update function
@parallel (1:size(Ey, 1) - 2, 1:size(Ey, 2), 1:size(Ey, 3) - 2) update_Ey!(Ey, dt, ε0, σ, Hx, Hz, dx, dz)

# Define the expected result manually based on the physics equation
expected_result_y = zeros(6, 5, 6)
for i in 1:4
    for j in 1:5
        for k in 1:4
            expected_result_y[i + 1, j, k + 1] = Ey[i + 1, j, k + 1] + dt / ε0 * (-σ * Ey[i, j, k] + (Hx[i, j, k + 1] - Hx[i, j, k]) - (Hz[i + 1, j, k] - Hz[i, j, k]) / dx)
        end
    end
end
@testset "Unit Test: update_Ey" begin
    @test isapprox(Ey, expected_result_y)
end

# # ---- Reference test ----
nt_ = 100
nx_ , ny_ , nz_ = 100, 100, 100
pml_width = 10
pml_alpha = 0.1

Hz = maxwell(nx_, ny_, nz_, nt_, pml_alpha; do_visu=false, do_test=true)

nx_pml, ny_pml, nz_pml = nx_ + 2 * pml_width, ny_ + 2 * pml_width, nz_ + 2 * pml_width

Hz_ref = zeros(Float32, nx_pml, ny_pml, nz_pml - 1)
Hz_ref = load("../test/ref_Hz_3D_cpu.jld")
array_value = Hz_ref["data"]

@testset "Reference Test: Hz_ref ≈ Hz" begin
    @test isapprox(Hz, array_value)
end