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

#@parallel_indices (i, j, k) function update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
function update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
    Ex[:, 2:end-1, 2:end-1] .+= dt / ε0 .* (-σ .* Ex[:, 2:end-1, 2:end-1] .+ diff(Hz, dims=2)./ dy .- diff(Hy, dims=3)./dz)
    #Ex[i, j, k] = Ex[i, j, k] + dt / ε0 * (-σ * Ex[i, j, k] + @d_ya(Hz) / dy - @d_za(Hy) / dz)
    return nothing    
end

function update_Ey!(Ey, dt, ε0, σ, Hx, Hz, dx, dz)
    Ey[2:end-1, :, 2:end-1] .+= dt / ε0 .* (-σ .* Ey[2:end-1, :, 2:end-1] .+ diff(Hx, dims=3)./dz .- diff(Hz, dims=1) ./ dx)
    return nothing
end

function update_Ez!(Ez, dt, ε0, σ, Hx, Hy, dx, dy)
    Ez[2:end-1, 2:end-1, :] .+= dt / ε0 .* (-σ .* Ez[2:end-1, 2:end-1, :] .+ diff(Hy, dims=1)./dx .- diff(Hx, dims=2) ./ dy)
    return nothing
end

function update_Hx!(Hx, dt, μ0, σ, Ey, Ez, dy, dz)
    Hx .+= dt / μ0 .* (-σ .* Hx .+ diff(Ey, dims=3)[2:end-1, :, :] ./ dz .- diff(Ez, dims=2)[2:end-1, :, :] ./ dy)
    return nothing
end

function update_Hy!(Hy, dt, μ0, σ, Ex, Ez, dx, dz)
    Hy .+= dt / μ0 .* (-σ .* Hy .+ diff(Ez, dims=1)[:, 2:end-1, :] ./ dx .- diff(Ex, dims=3)[:, 2:end-1, :] ./ dz)
    return nothing
end

function update_Hz!(Hz, dt, μ0, σ, Ex, Ey, dx, dy)
    Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2)[:, :, 2:end-1] ./ dy .- diff(Ey, dims=1)[:, :, 2:end-1] ./ dx)
    return nothing
end

function update_PML!(pml_width, pml_alpha, Ex, Ey, Ez)
    for i in 1:pml_width
        Ex[i, :, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[i, :, :]
        Ex[end - i + 1, :, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[end - i + 1, :, :]
        Ey[:, i, :] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, i, :]
        Ey[:, end - i + 1, :] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, end - i + 1 , :]
        Ez[:, :, i] .= exp(-(pml_width - i) * pml_alpha) .* Ez[:, :, i]
        Ez[:, :, end - i + 1] .= exp(-(pml_width - i) * pml_alpha) .* Ez[:, :, end - i + 1]
    end
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

    #println(size(Ex))

    Hx = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml-1, iy = 1:ny_pml, iz = 1:nz_pml])
    Hy = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml, iy = 1:ny_pml-1, iz = 1:nz_pml])
    Hz = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml, iy = 1:ny_pml, iz = 1:nz_pml-1])

    #Hx = [exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx-1, iy = 1:ny, iz = 1:nz]
    #Hy = [exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx, iy = 1:ny-1, iz = 1:nz]
    #Hz = [exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx, iy = 1:ny, iz = 1:nz-1]

    println("init ok")
    for it in 1:nt

        #println(size(Ex))
        #@parallel (1:size(Ex,1), 2:size(Ex, 2) - 2, 2:size(Ex, 3) - 2) update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
        update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
        #@synchronize()
        #println("ex ok")
        
        update_Ey!(Ey, dt, ε0, σ, Hx, Hz, dx, dz)
        #println("ey ok")

        update_Ez!(Ez, dt, ε0, σ, Hx, Hy, dx, dy)
        #println("ez ok")

        update_PML!(pml_width, pml_alpha, Ex, Ey, Ez)

        update_Hx!(Hx, dt, μ0, σ, Ey, Ez, dy, dz)
        #println("Hx ok")

        update_Hy!(Hy, dt, μ0, σ, Ex, Ez, dx, dz)
        #println("Hy ok")

        update_Hz!(Hz, dt, μ0, σ, Ex, Ey, dx, dy)
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
