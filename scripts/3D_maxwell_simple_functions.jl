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

function update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
    Ex[:, 2:end-1, 2:end-1] .+= dt / ε0 .* (-σ .* Ex[:, 2:end-1, 2:end-1] .+ diff(Hz, dims=2)./ dy .- diff(Hy, dims=3)./dz)
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

@views function maxwell()
    # physics
    lx, ly, lz = 40.0, 40.0, 40.0
    ε0 = 1.0
    μ0 = 1.0
    σ = 1.0
    # numerics
    nx, ny, nz = 100, 100, 100
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
    zc = LinRange(-lz / 2 + dz / 2, lz / 2 - dz / 2, nz)
    dt = min(dx, dy, dz)^2 / (1 / ε0 / μ0) / 4.1
    nt = 1000
    nout = 1e2
    # initial conditions
    Ex = @zeros(nx, ny + 1, nz + 1)
    Ey = @zeros(nx + 1, ny, nz + 1)
    Ez = @zeros(nx + 1, ny + 1, nz)
    
    Hx = @zeros(nx - 1, ny, nz)
    Hy = @zeros(nx, ny  - 1, nz)
    Hz = @zeros(nx, ny, nz - 1)

    Hx = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx-1, iy = 1:ny, iz = 1:nz])
    Hy = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx, iy = 1:ny-1, iz = 1:nz])
    Hz = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx, iy = 1:ny, iz = 1:nz-1])

    #Hx = [exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx-1, iy = 1:ny, iz = 1:nz]
    #Hy = [exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx, iy = 1:ny-1, iz = 1:nz]
    #Hz = [exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx, iy = 1:ny, iz = 1:nz-1]

    println("init ok")
    for it in 1:nt

        update_Ex!(Ex, dt, ε0, σ, Hy, Hz, dy, dz)
        #println("ex ok")
        
        update_Ey!(Ey, dt, ε0, σ, Hx, Hz, dx, dz)
        #println("ey ok")

        update_Ez!(Ez, dt, ε0, σ, Hx, Hy, dx, dy)
        #println("ez ok")

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
