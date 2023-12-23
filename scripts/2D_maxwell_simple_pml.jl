using Plots

@views function maxwell()
    # physics
    lx, ly = 40.0, 40.0
    ε0 = 1.0
    μ0 = 1.0
    σ = 1.0
    # numerics
    
    nx, ny = 100, 101

    # PML parameters
    pml_width = 10
    pml_alpha = 0.1
     
    # Extend the grid
    nx_pml, ny_pml = nx + 2 * pml_width, ny + 2 * pml_width
    #println("nx_pml: ", nx_pml)
    #println("ny_pml: ", ny_pml) 

    dx, dy = lx / nx_pml, ly / ny_pml
    xc, yc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx_pml), LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny_pml)
    dt = min(dx, dy)^2 / (1 / ε0 / μ0) / 4.1
    nt = 1e3
    nout = 1e2




    # initial conditions
    Ex = zeros(nx_pml, ny_pml + 1)
    Ey = zeros(nx_pml + 1, ny_pml)
    Hz = zeros(nx_pml, ny_pml)
    Hz = exp.(.-xc .^ 2 .- yc' .^ 2)
    for it in 1:nt
        # Update PML
        for i in 1:pml_width
            Ex[i, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[i, :]
            Ex[end - i + 1, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[end - i + 1, :]
            Ey[:, i] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, i]
            Ey[:, end - i + 1] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, end - i + 1]
        end
        # println("update pml okay")

        # println("Hz: ", size(Hz))
        # println("Ex: ",size(Ex))
        # println("Ey: ",size(Ey))

        # println("1 ", size(Ex[:, 2:end-1]))
        # println("2 ",size(-σ .* Ex[:, 2:end-1]))
        # println("3 ",size(diff(Hz, dims=2)./ dy))
        

        Ex[:, 2:end-1] .+= dt / ε0 .* (-σ .* Ex[:, 2:end-1] .+ diff(Hz, dims=2)./ dy)
        #println("update Ex okay")
        
        Ey[2:end-1, :] .+= dt / ε0 .* (-σ .* Ey[2:end-1, :] .- diff(Hz, dims=1) ./ dx)
        Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2) ./ dy .- diff(Ey, dims=1) ./ dx)
       

        #println("updates okay")
        (it % nout == 0) && display(heatmap(Hz'))
    end
    return
end

maxwell()
