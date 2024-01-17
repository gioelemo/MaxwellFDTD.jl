using Plots

@views function maxwell()
    # physics
    lx, ly = 40.0, 40.0
    ε0 = 1.0
    μ0 = 1.0
    σ = 1.0
    # numerics
    nx, ny = 100, 101
    dx, dy = lx / nx, ly / ny
    xc, yc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx), LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
    dt = min(dx, dy)^2 / (1 / ε0 / μ0) / 4.1
    nt = 1e3
    nout = 1e2
    # initial conditions
    Ex = zeros(nx, ny + 1)
    Ey = zeros(nx + 1, ny)
    Hz = zeros(nx, ny)
    Hz = exp.(.-xc .^ 2 .- yc' .^ 2)
    for it in 1:nt
        Ex[:, 2:end-1] .+= dt / ε0 .* (-σ .* Ex[:, 2:end-1] .+ diff(Hz, dims=2) ./ dy)
        Ey[2:end-1, :] .+= dt / ε0 .* (-σ .* Ey[2:end-1, :] .- diff(Hz, dims=1) ./ dx)
        Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2) ./ dy .- diff(Ey, dims=1) ./ dx)
        (it % nout == 0) && display(heatmap(Hz', color=:turbo, aspect_ratio=:equal, xlims=(0, nx), ylims=(0, ny), legend=false))
    end
    return
end

maxwell()
