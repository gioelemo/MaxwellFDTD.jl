const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
end

using Plots, JLD

"""
    save_array(Aname,A)

Store an array in a binary format (used then for plotting)
"""
function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end

"""
    compute_d_xa!(A, dA)

Helper (parallel) function to compute the derivative of a matrix A 
in the dimension x and store the result into dA
"""
@parallel function compute_d_xa!(A, dA)
    @all(dA) = @d_xa(A)
    return nothing
end

"""
    compute_d_ya!(A, dA)

Helper (parallel) function to compute the derivative of a matrix A 
in the dimension y and store the result into dA
"""
@parallel function compute_d_ya!(A, dA)
    @all(dA) = @d_ya(A)
    return nothing
end

"""
    compute_d_za!(A, dA)

Helper (parallel) function to compute the derivative of a matrix A 
in the dimension z and store the result into dA
"""
@parallel function compute_d_za!(A, dA)
    @all(dA) = @d_za(A)
    return nothing
end

"""
    update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)

Update the Ex field
"""
@parallel_indices (i, j, k) function update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
    Ex[i, j + 1, k + 1] = Ex[i, j + 1, k + 1] + dt / ε0 * (-σ * Ex[i, j + 1, k + 1] + @d_ya(Hz) / dy - @d_za(Hy) / dz)
    return nothing    
end

"""
    update_Ey!(Ey, dt, ε0, σ, Hx, Hz, dx, dz)

Update the Ey field
"""
@parallel_indices (i, j, k) function update_Ey!(Ey, dt, ε0, σ, Hx, Hz, dx, dz)
    Ey[i + 1, j, k + 1] = Ey[i + 1, j, k + 1] + dt / ε0 * (-σ * Ey[i + 1, j, k + 1] + @d_za(Hx) / dz - @d_xa(Hz) / dx)
    return nothing
end

"""
    update_Ez!(Ez, dt, ε0, σ, Hx, Hy, dx, dy)

Update the Ez field
"""
@parallel_indices (i, j, k) function update_Ez!(Ez, dt, ε0, σ, Hx, Hy, dx, dy)
    Ez[i + 1, j + 1, k] = Ez[i + 1, j + 1, k] + dt / ε0 * (-σ * Ez[i + 1, j + 1, k]  + @d_xa(Hy) / dx - @d_ya(Hx) / dy)
    return nothing
end

"""
    update_Hx!(Hx, dt, μ0, σ, Ey, dz_Ey, Ez, dy_Ez, dy, dz)

Update the Hx field
"""
@parallel_indices (i, j, k) function update_Hx!(Hx, dt, μ0, σ, Ey, dz_Ey, Ez, dy_Ez, dy, dz)
    Hx[i, j, k] = Hx[i, j, k] + dt / μ0 * (-σ * Hx[i, j, k] + dz_Ey[i + 1, j, k] / dz - dy_Ez[i + 1, j, k] / dy)
    return nothing
end

"""
    update_Hy!(Hy, dt, μ0, σ, Ex, dz_Ex, Ez, dx_Ez, dx, dz)

Update the Hy field
"""
@parallel_indices (i, j, k) function update_Hy!(Hy, dt, μ0, σ, Ex, dz_Ex, Ez, dx_Ez, dx, dz)
    Hy[i, j, k] = Hy[i, j, k] + dt / μ0 * (-σ * Hy[i, j, k] + dx_Ez[i, j + 1, k] / dx - dz_Ex[i, j + 1, k] / dz)
    return nothing
end

"""
    update_Hz!(Hz, dt, μ0, σ, Ex, dy_Ex, Ey, dx_Ey, dx, dy)

Update the Hz field
"""
@parallel_indices (i, j, k) function update_Hz!(Hz, dt, μ0, σ, Ex, dy_Ex, Ey, dx_Ey, dx, dy)
    Hz[i, j, k] = Hz[i, j, k] + dt / μ0 * (-σ * Hz[i, j, k] + dy_Ex[i, j, k + 1] / dy - dx_Ey[i, j, k + 1] / dx)
    return nothing
end

"""
    update_PML_x!(pml_width, pml_alpha, Ex)

Update the PML for the Ex field in the x-direction
"""
@parallel_indices (i,j,k) function update_PML_x!(pml_width, pml_alpha, Ex)
    Ex[i, j, k] = exp(-(pml_width - i) * pml_alpha) * Ex[i, j, k]
    Ex[end - i + 1, j, k] = exp(-(pml_width - i) * pml_alpha) * Ex[end - i + 1, j, k]
    return nothing
end

"""
    update_PML_y!(pml_width, pml_alpha, Ey)

Update the PML for the Ey field in the y-direction
"""
@parallel_indices (i,j,k) function update_PML_y!(pml_width, pml_alpha, Ey)
    Ey[i, j, k] = exp(-(pml_width - j) * pml_alpha) * Ey[i, j, k]
    Ey[i, end - j + 1, k] = exp(-(pml_width - j) * pml_alpha) * Ey[i, end - j + 1 , k]
    return nothing
end

"""
    update_PML_z!(pml_width, pml_alpha, Ez)

Update the PML for the Ez field in the z-direction
"""
@parallel_indices (i,j,k) function update_PML_z!(pml_width, pml_alpha, Ez)
    Ez[i, j, k] = exp(-(pml_width - k) * pml_alpha) * Ez[i, j, k]
    Ez[i, j, end - k + 1] = exp(-(pml_width - k) * pml_alpha) * Ez[i, j, end - k + 1]
    return nothing
end

"""
    maxwell(nx_, ny_, nz_, nt_, pml_alpha_; do_visu=false, do_check=true, do_test=true)

Use the Finite Difference Time Domain (FDTD) solver to solve Maxwell's equations

# Arguments
- `nx_::Integer`: Number of x discretization-steps.
- `ny_::Integer`: Number of y discretization-steps.
- `nz_::Integer`: Number of z discretization-steps.
- `nt_::Integer`: Number of timesteps.
- `pml_alpha_::Float` : "Strength" of the PML layer
- `do_visu::Boolean=false`: Perform visualisation.
- `do_test::Boolean=false`: Perform testing (generate a jld reference file).
"""
@views function maxwell(nx_, ny_, nz_, nt_, pml_alpha_; do_visu=false, do_test=true)
    # Physics
    lx, ly, lz = 40.0, 40.0, 40.0   # physical size
    ε0 = 1.0                        # permittivity
    μ0 = 1.0                        # permeability
    σ = 1.0                         # electrical conductivity
    
    # Numerics
    nx, ny, nz = nx_, ny_, nz_      # number space steps

    # PML parameters
    pml_width = 10                  # PML extensions
    pml_alpha = pml_alpha_          # PML "strength"
    
    # Extend the grid
    nx_pml, ny_pml, nz_pml = nx + 2 * pml_width, ny + 2 * pml_width, nz + 2 * pml_width
    
    # Other numerics parameters
    dx, dy, dz = lx / nx_pml, ly / ny_pml, lz / nz_pml
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx_pml)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny_pml)
    zc = LinRange(-lz / 2 + dz / 2, lz / 2 - dz / 2, nz_pml)
    dt = min(dx, dy, dz)^2 / (1 / ε0 / μ0) / 4.1
    nt = nt_

    # Initial conditions
    # E-fields
    Ex = @zeros(nx_pml, ny_pml + 1, nz_pml + 1)
    Ey = @zeros(nx_pml + 1, ny_pml, nz_pml + 1)
    Ez = @zeros(nx_pml + 1, ny_pml + 1, nz_pml)
    
    # H-fields
    Hx = @zeros(nx_pml - 1, ny_pml, nz_pml)
    Hy = @zeros(nx_pml, ny_pml  - 1, nz_pml)
    Hz = @zeros(nx_pml, ny_pml, nz_pml - 1)

    # Gaussian initial condition
    Hx = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml-1, iy = 1:ny_pml, iz = 1:nz_pml])
    Hy = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml, iy = 1:ny_pml-1, iz = 1:nz_pml])
    Hz = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml, iy = 1:ny_pml, iz = 1:nz_pml-1])

    # Derivative matrix
    dy_Ez = @zeros(nx_pml + 1, ny_pml, nz_pml)
    dz_Ey = @zeros(nx_pml + 1, ny_pml, nz_pml)
    dz_Ex = @zeros(nx_pml, ny_pml + 1, nz_pml)
    dx_Ez = @zeros(nx_pml, ny_pml + 1, nz_pml)
    dy_Ex = @zeros(nx_pml, ny_pml, nz_pml + 1)
    dx_Ey = @zeros(nx_pml, ny_pml, nz_pml + 1)

    #println("init ok")
    
    # Timestepping
    for it in 1:nt

        # Update Ex field
        @parallel (1:size(Ex, 1), 1:size(Ex, 2)-2, 1:size(Ex, 3) - 2) update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
        
        # Update Ey field
        @parallel (1:size(Ey, 1) - 2, 1:size(Ey, 2), 1:size(Ey, 3) - 2) update_Ey!(Ey, dt, ε0, σ, Hx, Hz, dx, dz)

        # Update Ez field
        @parallel (1:size(Ez, 1) - 2, 1:size(Ez, 2) - 2, 1:size(Ez, 3)) update_Ez!(Ez, dt, ε0, σ, Hx, Hy, dx, dy)

        # Update PML
        if pml_width > 0
            @parallel (1:pml_width, 1:size(Ex, 2), 1:size(Ex,3)) update_PML_x!(pml_width, pml_alpha, Ex)
            @parallel (1:size(Ey, 1), 1:pml_width, 1:size(Ey,3)) update_PML_y!(pml_width, pml_alpha, Ey)
            @parallel (1:size(Ez, 1), 1:size(Ez,2), 1:pml_width) update_PML_z!(pml_width, pml_alpha, Ez)
        end

        # Compute derivative and update Hx field
        @parallel compute_d_ya!(Ez, dy_Ez) 
        @parallel compute_d_za!(Ey, dz_Ey)
        @parallel (1:size(Hx, 1), 1:size(Hx, 2), 1:size(Hx, 3)) update_Hx!(Hx, dt, μ0, σ, Ey, dz_Ey, Ez, dy_Ez, dy, dz)

        # Compute derivative and update Hy field
        @parallel compute_d_za!(Ex, dz_Ex)
        @parallel compute_d_xa!(Ez, dx_Ez)
        @parallel (1:size(Hy, 1), 1:size(Hy, 2), 1:size(Hy, 3)) update_Hy!(Hy, dt, μ0, σ, Ex, dz_Ex, Ez, dx_Ez, dx, dz)

        # Compute derivative and update Hy field
        @parallel compute_d_ya!(Ex, dy_Ex)
        @parallel compute_d_xa!(Ey, dx_Ey)
        @parallel (1:size(Hz, 1), 1:size(Hz, 2), 1:size(Hz, 3)) update_Hz!(Hz, dt, μ0, σ, Ex, dy_Ex, Ey, dx_Ey, dx, dy)
        
        println(it)
    end

    # Visualisation
    if do_visu == true
        # Save the final fields into array
        save_array("../docs/out_Ex",convert.(Float32,Array(Ex)))
        save_array("../docs/out_Ey",convert.(Float32,Array(Ey)))
        save_array("../docs/out_Ez",convert.(Float32,Array(Ez)))

        save_array("../docs/out_Hx",convert.(Float32,Array(Hx)))
        save_array("../docs/out_Hy",convert.(Float32,Array(Hy)))
        save_array("../docs/out_Hz",convert.(Float32,Array(Hz)))
    end

    # Testing
    if do_test == true
        if USE_GPU
            save("../test/ref_Hz_3D_gpu.jld", "data", Hz)         # store case for reference testing
        else
            save("../test/ref_Hz_3D_cpu.jld", "data", Hz)         # store case for reference testing
        end
    end

    return Array(Hz)
end

#maxwell()

#maxwell(nx_, ny_, nz_, nt_, pml_alpha_; do_visu=false, do_check=true, do_test=true)
#maxwell(100, 100, 100, 1000, 0.1; do_visu=true, do_test=false)

#maxwell(100, 100, 100, 100, 0.1; do_visu=true, do_test=true)

#maxwell(256, 256, 100, 15000, 0.0, do_visu=true, do_test=false)