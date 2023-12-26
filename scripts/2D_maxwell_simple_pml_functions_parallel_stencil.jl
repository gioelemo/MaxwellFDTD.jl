const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=true)
end

using Plots
plot_font = "Computer Modern"
default(fontfamily=plot_font, framestyle=:box, label=true, grid=true, labelfontsize=11, tickfontsize=11, titlefontsize=13)

@parallel function update_Ex!(Ex, Hz, σ, ε0, dt, dy)
    #Ex[:, 2:end-1] .+= dt / ε0 .* (-σ .* Ex[:, 2:end-1] .+ diff(Hz, dims=2)./ dy)
    @inn_y(Ex) = @inn_y(Ex) .+ dt / ε0 .* (-σ .* @inn_y(Ex) .+ @d_ya(Hz) ./ dy)
    return nothing
end

@parallel function update_Ey!(Ey, Hz, σ, ε0, dt, dx)
    #Ey[2:end-1, :] .+= dt / ε0 .* (-σ .* Ey[2:end-1, :] .- diff(Hz, dims=1) ./ dx)
    @inn_x(Ey) = @inn_x(Ey) .+ dt / ε0 .* (-σ .* @inn_x(Ey) .- @d_xa(Hz) ./ dx)
    return nothing
end

@parallel_indices (i) function update_PML!(pml_width, pml_alpha, Ex, Ey)
    # for i in 1:pml_width
    #     Ex[i, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[i, :]
    #     Ex[end - i + 1, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[end - i + 1, :]
    #     Ey[:, i] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, i]
    #     Ey[:, end - i + 1] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, end - i + 1]
    # end
    Ex[i, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[i, :]
    Ex[end - i + 1, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[end - i + 1, :]
    Ey[:, i] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, i]
    Ey[:, end - i + 1] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, end - i + 1]

    return nothing
end

@parallel function update_Hz!(Hz, Ex, Ey, σ, μ0, dt, dy, dx)
    #Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2) ./ dy .- diff(Ey, dims=1) ./ dx)
    @all(Hz) = @all(Hz) .+ dt / μ0 .* (-σ .* @all(Hz) .+ @d_ya(Ex) ./ dy .- @d_xa(Ey) ./ dx)
    return nothing
end
    

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
    pml_alpha = 0
     
    # Extend the grid
    nx_pml, ny_pml = nx + 2 * pml_width, ny + 2 * pml_width

    # Other numerics parameters
    dx, dy = lx / nx_pml, ly / ny_pml
    xc, yc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx_pml), LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny_pml)
    dt = min(dx, dy)^2 / (1 / ε0 / μ0) / 4.1
    nt = 1e3
    nout = 1e2

    # initial conditions
    Ex = @zeros(nx_pml, ny_pml + 1)
    Ey = @zeros(nx_pml + 1, ny_pml)
    Hz = @zeros(nx_pml, ny_pml)
    Hz = Data.Array(exp.(.-xc .^ 2 .- yc' .^ 2))

    for it in 1:nt
        # Update E
        @parallel update_Ex!(Ex, Hz, σ, ε0, dt, dy)
        @parallel update_Ey!(Ey, Hz, σ, ε0, dt, dx)

        # Update PML
        @parallel (1:pml_width) update_PML!(pml_width, pml_alpha, Ex, Ey)
        
        # Update H
        @parallel update_Hz!(Hz, Ex, Ey, σ, μ0, dt, dy, dx)
        
        # Save the final field
        save_end_ez = false
       
        if it % nout == 0
            # Create a heatmap
            plt = heatmap(Hz', aspect_ratio=:equal, xlims=(1, nx_pml), ylims=(1, ny_pml), c=:turbo, title="H_z field")

            # Add a rectangle to represent the PML layer
            rect_x = [pml_width, nx_pml-pml_width+1, nx_pml-pml_width+1, pml_width, pml_width ]
            rect_y = [pml_width, pml_width, ny_pml-pml_width+1, nx_pml-pml_width+1, pml_width]
            plot!(plt, rect_x, rect_y, line=:black, linewidth=2, fillalpha=0, legend=false)
            
            # Save the figure
            if it ==nt && save_end_ez==true
                savefig(plt, "maxwell_pml.png")
            end
            # Display the plot
            display(plt)
        end
    end
    return
end

maxwell()
