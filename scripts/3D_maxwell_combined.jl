const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
end

using Plots

"""
    save_array(Aname,A)

Store an array in a binary format (used then for plotting)
"""
function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end

@parallel function compute_d_xa!(A, dA)
    @all(dA) = @d_xa(A)
    return nothing
end

@parallel function compute_d_ya!(A, dA)
    @all(dA) = @d_ya(A)
    return nothing
end

@parallel function compute_d_za!(A, dA)
    @all(dA) = @d_za(A)
    return nothing
end

@parallel_indices (i, j, k) function update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
    Ex[i, j + 1, k + 1] = Ex[i, j + 1, k + 1] + dt / ε0 * (-σ * Ex[i, j + 1, k + 1] + @d_ya(Hz) / dy - @d_za(Hy) / dz)
    return nothing    
end

@parallel_indices (i, j, k) function update_Ey!(Ey, dt, ε0, σ, Hx, Hz, dx, dz)
    Ey[i + 1, j, k + 1] = Ey[i + 1, j, k + 1] + dt / ε0 * (-σ * Ey[i + 1, j, k + 1] + @d_za(Hx) / dz - @d_xa(Hz) / dx)
    return nothing
end

@parallel_indices (i, j, k) function update_Ez!(Ez, dt, ε0, σ, Hx, Hy, dx, dy)
    Ez[i + 1, j + 1, k] = Ez[i + 1, j + 1, k] + dt / ε0 * (-σ * Ez[i + 1, j + 1, k]  + @d_xa(Hy) / dx - @d_ya(Hx) / dy)
    return nothing
end

@parallel_indices (i, j, k) function update_Hx!(Hx, dt, μ0, σ, Ey, dz_Ey, Ez, dy_Ez, dy, dz)
    Hx[i, j, k] = Hx[i, j, k] + dt / μ0 * (-σ * Hx[i, j, k] + dz_Ey[i + 1, j, k] / dz - dy_Ez[i + 1, j, k] / dy)
    return nothing
end

@parallel_indices (i, j, k) function update_Hy!(Hy, dt, μ0, σ, Ex, dz_Ex, Ez, dx_Ez, dx, dz)
    Hy[i, j, k] = Hy[i, j, k] + dt / μ0 * (-σ * Hy[i, j, k] + dx_Ez[i, j + 1, k] / dx - dz_Ex[i, j + 1, k] / dz)
    return nothing
end

@parallel_indices (i, j, k) function update_Hz!(Hz, dt, μ0, σ, Ex, dy_Ex, Ey, dx_Ey, dx, dy)
    Hz[i, j, k] = Hz[i, j, k] + dt / μ0 * (-σ * Hz[i, j, k] + dy_Ex[i, j, k + 1] / dy - dx_Ey[i, j, k + 1] / dx)
    return nothing
end

@parallel_indices (i,j,k) function update_PML_x!(pml_width, pml_alpha, Ex)
    Ex[i, j, k] = exp(-(pml_width - i) * pml_alpha) * Ex[i, j, k]
    Ex[end - i + 1, j, k] = exp(-(pml_width - i) * pml_alpha) * Ex[end - i + 1, j, k]
    return nothing
end

@parallel_indices (i,j,k) function update_PML_y!(pml_width, pml_alpha, Ey)
    Ey[i, j, k] = exp(-(pml_width - j) * pml_alpha) * Ey[i, j, k]
    Ey[i, end - j + 1, k] = exp(-(pml_width - j) * pml_alpha) * Ey[i, end - j + 1 , k]
    return nothing
end

@parallel_indices (i,j,k) function update_PML_z!(pml_width, pml_alpha, Ez)
    Ez[i, j, k] = exp(-(pml_width - k) * pml_alpha) * Ez[i, j, k]
    Ez[i, j, end - k + 1] = exp(-(pml_width - k) * pml_alpha) * Ez[i, j, end - k + 1]
    return nothing
end

@views function maxwell()
    # physics
    lx, ly, lz = 40.0, 40.0, 40.0
    ε0 = 1.0
    μ0 = 1.0
    σ = 1.0
    pml_width = 10
    pml_alpha = 0.1

    # numerics
    nx, ny, nz = 100, 100, 100
    
    # Extend the grid
    nx_pml, ny_pml, nz_pml = nx + 2 * pml_width, ny + 2 * pml_width, nz + 2 * pml_width
    
    dx, dy, dz = lx / nx_pml, ly / ny_pml, lz / nz_pml
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx_pml)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny_pml)
    zc = LinRange(-lz / 2 + dz / 2, lz / 2 - dz / 2, nz_pml)
    dt = min(dx, dy, dz)^2 / (1 / ε0 / μ0) / 4.1
    nt = 1000
    nout = 1e2

    # initial conditions
    Ex = @zeros(nx_pml, ny_pml + 1, nz_pml + 1)
    Ey = @zeros(nx_pml + 1, ny_pml, nz_pml + 1)
    Ez = @zeros(nx_pml + 1, ny_pml + 1, nz_pml)
    
    Hx = @zeros(nx_pml - 1, ny_pml, nz_pml)
    Hy = @zeros(nx_pml, ny_pml  - 1, nz_pml)
    Hz = @zeros(nx_pml, ny_pml, nz_pml - 1)

    Hx = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml-1, iy = 1:ny_pml, iz = 1:nz_pml])
    Hy = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml, iy = 1:ny_pml-1, iz = 1:nz_pml])
    Hz = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml, iy = 1:ny_pml, iz = 1:nz_pml-1])

    #Hx = [exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx-1, iy = 1:ny, iz = 1:nz]
    #Hy = [exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx, iy = 1:ny-1, iz = 1:nz]
    #Hz = [exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx, iy = 1:ny, iz = 1:nz-1]

    dy_Ez = @zeros(nx_pml + 1, ny_pml, nz_pml)
    dz_Ey = @zeros(nx_pml + 1, ny_pml, nz_pml)
    dz_Ex = @zeros(nx_pml, ny_pml + 1, nz_pml)
    dx_Ez = @zeros(nx_pml, ny_pml + 1, nz_pml)
    dy_Ex = @zeros(nx_pml, ny_pml, nz_pml + 1)
    dx_Ey = @zeros(nx_pml, ny_pml, nz_pml + 1)

    println("init ok")
    for it in 1:nt

        # Update Ex field
        @parallel (1:size(Ex, 1), 1:size(Ex, 2)-2, 1:size(Ex, 3) - 2) update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
        #println("ex ok")
        
        # Update Ey field
        @parallel (1:size(Ey, 1) - 2, 1:size(Ey, 2), 1:size(Ey, 3) - 2) update_Ey!(Ey, dt, ε0, σ, Hx, Hz, dx, dz)
        #println("ey ok")

        # Update Ez field
        @parallel (1:size(Ez, 1) - 2, 1:size(Ez, 2) - 2, 1:size(Ez, 3)) update_Ez!(Ez, dt, ε0, σ, Hx, Hy, dx, dy)
        #println("ez ok")

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
        #println("Hx ok")

        # Compute derivative and update Hy field
        @parallel compute_d_za!(Ex, dz_Ex)
        @parallel compute_d_xa!(Ez, dx_Ez)
        @parallel (1:size(Hy, 1), 1:size(Hy, 2), 1:size(Hy, 3)) update_Hy!(Hy, dt, μ0, σ, Ex, dz_Ex, Ez, dx_Ez, dx, dz)
        #println("Hy ok")

        # Compute derivative and update Hy field
        @parallel compute_d_ya!(Ex, dy_Ex)
        @parallel compute_d_xa!(Ey, dx_Ey)
        @parallel (1:size(Hz, 1), 1:size(Hz, 2), 1:size(Hz, 3)) update_Hz!(Hz, dt, μ0, σ, Ex, dy_Ex, Ey, dx_Ey, dx, dy)
        #println("Hz ok")
        
        println(it)
    end

    save_array("../docs/out_Ex",convert.(Float32,Array(Ex)))
    save_array("../docs/out_Ey",convert.(Float32,Array(Ey)))
    save_array("../docs/out_Ez",convert.(Float32,Array(Ez)))

    save_array("../docs/out_Hx",convert.(Float32,Array(Hx)))
    save_array("../docs/out_Hy",convert.(Float32,Array(Hy)))
    save_array("../docs/out_Hz",convert.(Float32,Array(Hz)))

    return
end

maxwell()
