const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=true)
end

using Printf, Plots, JLD
plot_font = "Computer Modern"
default(fontfamily=plot_font, framestyle=:box, label=true, grid=true, labelfontsize=11, tickfontsize=11, titlefontsize=13)

"""
    update_Ex!(Ex, Hz, σ, ε0, dt, dy)

Update the Ex field
"""
@parallel function update_Ex!(Ex, Hz, σ, ε0, dt, dy)
    @inn_y(Ex) = @inn_y(Ex) + dt / ε0 * (-σ * @inn_y(Ex) + @d_ya(Hz) / dy)
    return nothing
end

"""
    update_Ey!(Ey, Hz, σ, ε0, dt, dx)

Update the Ey field
"""
@parallel function update_Ey!(Ey, Hz, σ, ε0, dt, dx)
    @inn_x(Ey) = @inn_x(Ey) + dt / ε0 * (-σ * @inn_x(Ey) - @d_xa(Hz) / dx)
    return nothing
end

"""
    update_PML_x!(pml_width, pml_alpha, Ex)

Update the x-regions of the pml
"""
@parallel_indices (i,j) function update_PML_x!(pml_width, pml_alpha, Ex)
    Ex[i, j] = exp(-(pml_width - i) * pml_alpha) * Ex[i, j]
    Ex[end - i + 1, j] = exp(-(pml_width - i) * pml_alpha) * Ex[end - i + 1, j]
    return nothing
end

"""
    update_PML_y!(pml_width, pml_alpha, Ey)

Update the y-regions of the pml
"""
@parallel_indices (i,j) function update_PML_y!(pml_width, pml_alpha, Ey)
    Ey[j, i] = exp(-(pml_width - i) * pml_alpha) * Ey[j, i]
    Ey[j, end - i + 1] = exp(-(pml_width - i) * pml_alpha) * Ey[j, end - i + 1]
    return nothing
end

"""
    update_Hz!(Hz, Ex, Ey, σ, μ0, dt, dy, dx)

Update the Hz field
"""
@parallel function update_Hz!(Hz, Ex, Ey, σ, μ0, dt, dy, dx)
    @all(Hz) = @all(Hz) + dt / μ0 * (-σ * @all(Hz) + @d_ya(Ex) / dy - @d_xa(Ey) / dx)
    return nothing
end
    
"""
    maxwell(ny_, nt_, nvis_, pml_alpha_; do_visu=false, do_test=true)

Use the Finite Difference Time Domain (FDTD) solver to solve Maxwell's equations

# Arguments
- `ny_::Integer`: Number of y discretization-steps.
- `nt_::Integer`: Number of timesteps.
- `nvis_::Integer`: Number of steps between visualisation output.
- `pml_alpha_::Float` : "Strength" of the PML layer
- `do_visu::Boolean=false`: Perform visualisation.
- `do_test::Boolean=false`: Perform testing (generate a jld reference file).
"""
@views function maxwell(ny_, nt_, nvis_, pml_alpha_; do_visu=false, do_test=true)
    # Physics
    lx, ly = 40.0, 40.0       # physical size
    ε0 = 1.0                  # permittivity
    μ0 = 1.0                  # permeability
    σ = 1.0                   # electrical conductivity
    
    # Numerics
    nx, ny = ny_ - 1, ny_     # number space steps

    # PML parameters
    pml_width = 50            # PML extensions
    pml_alpha = pml_alpha_    # PML "strength"
     
    # Extend the grid
    nx_pml, ny_pml = nx + 2 * pml_width, ny + 2 * pml_width

    # Other numerics parameters
    dx, dy = lx / nx_pml, ly / ny_pml
    xc, yc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx_pml), LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny_pml)
    dt = min(dx, dy)^2 / (1 / ε0 / μ0) / 4.1
    nt = nt_
    nout = nvis_

    # Initial conditions
    # E-fields
    Ex = @zeros(nx_pml, ny_pml + 1)
    Ey = @zeros(nx_pml + 1, ny_pml)

    # H-fields
    Hz = @zeros(nx_pml, ny_pml)
    Hz = Data.Array(exp.(.-xc .^ 2 .- yc' .^ 2))

    # Visualisation for cluster
    if do_visu
        # plotting environment
        ENV["GKSwstype"]="nul"
        if isdir("../docs/viz_out_2D")==false mkdir("../docs/viz_out_2D") end
        loadpath = "../docs/viz_out_2D/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        iframe = 0
    end

    # Timestepping
    for it in 1:nt
        # Update E
        @parallel update_Ex!(Ex, Hz, σ, ε0, dt, dy)
        @parallel update_Ey!(Ey, Hz, σ, ε0, dt, dx)

        # Update PML
        if pml_width > 0
            @parallel (1:pml_width, 1:size(Ex, 2)) update_PML_x!(pml_width, pml_alpha, Ex)
            @parallel (1:pml_width, 1:size(Ey, 1)) update_PML_y!(pml_width, pml_alpha, Ey)
        end
        
        # Update H
        @parallel update_Hz!(Hz, Ex, Ey, σ, μ0, dt, dy, dx)
        
        # Visualisation
        if it % nout == 0 && do_visu == true
            # Create a heatmap
            plt = heatmap(Array(Hz'), aspect_ratio=:equal, xlims=(1, nx_pml), ylims=(1, ny_pml), c=:turbo, title="\$H_z\$ at it=$it")

            # Add a rectangle to represent the PML layer
            rect_x = [pml_width, nx_pml-pml_width+1, nx_pml-pml_width+1, pml_width, pml_width ]
            rect_y = [pml_width, pml_width, ny_pml-pml_width+1, nx_pml-pml_width+1, pml_width]
            plot!(plt, rect_x, rect_y, line=:black, linewidth=2, fillalpha=0, legend=false)

            png(plt, @sprintf("../docs/viz_out_2D/maxwell2D_%04d.png",iframe+=1))
            
            # Display the plot (work only local)
            # display(plt)
        end
    end

    # Testing
    if do_test == true
        if USE_GPU
            save("../test/ref_Hz_2D_gpu.jld", "data", Hz)         # store case for reference testing
        else
            save("../test/ref_Hz_2D_cpu.jld", "data", Hz)         # store case for reference testing
        end
    end

    return Array(Hz)
end

# ny, nt, nvis, pml_alpha

# Functions used for testing
#maxwell(50, 10, 10, 0.25; do_visu=false, do_test=true)
#maxwell(50, 10, 10, 0.25; do_visu=true, do_test=false)

# Function used for the simulations in README.md
#maxwell(256, 15000, 100, 0.0; do_visu=true, do_test=false)
#maxwell(256, 15000, 100, 5.0; do_visu=true, do_test=false)
#maxwell(256, 15000, 100, 0.1; do_visu=true, do_test=false)

