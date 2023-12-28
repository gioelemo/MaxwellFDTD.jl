const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 1, inbounds=true)
end
using Plots, LaTeXStrings, Colors, ColorSchemes, Printf, JLD

plot_font = "Computer Modern"
default(fontfamily=plot_font, framestyle=:box, label=true, grid=true, labelfontsize=11, tickfontsize=11, titlefontsize=13)

"""
    update_H_y!(H_y, E_z, H_y_e_loss, H_y_h_loss)

Update the value of the Hy field.
"""
@parallel_indices (i) function update_H_y!(H_y, E_z, H_y_e_loss, H_y_h_loss)
    n = length(H_y)
    if i < n
        H_y[i] = H_y_h_loss[i] * H_y[i] + H_y_e_loss[i] * (E_z[i+1] - E_z[i]) 
    end
    return nothing
end

"""
    update_E_z!(H_y, E_z, E_z_e_loss, E_z_h_loss)

Update the value of the Ez field.
"""
@parallel_indices (i) function update_E_z!(H_y, E_z, E_z_e_loss, E_z_h_loss) 
    if i > 1
         E_z[i] = E_z_e_loss[i] * E_z[i] + E_z_h_loss[i] * (H_y[i] - H_y[i-1])
    end
    return nothing
end

"""
    update_E_z_loss_coeff!()

Update the value of the `Ez` coefficient-loss field depending on the `x` position (with loss layer)

# Arguments
- `E_z_e_loss::Float`: Loss of the E_z field due to E.
- `E_z_h_loss::Float`: Loss of the E_z field due to H.
- `imp0::Float` : Impedence of the free space (usually 377.0 Ω)
- `loss::Float` : Loss value introduced after `loss_layer_index`
- `interface_index::Integer` : Index of the `x` position where interface between free and dielectric space is introduced
- `loss_layer_index::Integer` : Index of the `x` position where the loss is introduced
- `epsR::Float` : Relative permittivity of the dielectric region
"""
@parallel_indices (i) function update_E_z_loss_coeff!(E_z_e_loss, E_z_h_loss, imp0, loss, interface_index, loss_layer_index, epsR)
    if i < interface_index
        E_z_e_loss[i] = 1.0
        E_z_h_loss[i] = imp0
    elseif i < loss_layer_index
        E_z_e_loss[i] = 1.0
        E_z_h_loss[i] = imp0 / epsR
    else
        E_z_e_loss[i] = (1.0 - loss) / (1.0 + loss)
        E_z_h_loss[i] = imp0 / epsR  / (1.0 + loss)
    end
    return nothing
end

"""
    update_E_z_loss_coeff!()

Update the value of the `Ez` coefficient-loss field depending on the `x` position (without loss layer)

# Arguments
- `E_z_e_loss::Float`: Loss of the `Ez` field due to `E`.
- `E_z_h_loss::Float`: Loss of the `E_z` field due to `H`.
- `imp0::Float` : Impedence of the free space (usually 377.0 Ω)
- `loss::Float` : Loss value introduced after `loss_layer_index`
- `interface_index::Integer` : Index of the `x` position where interface between free and dielectric space is introduced
- `epsR::Float` : Relative permittivity of the dielectric region
"""
@parallel_indices (i) function update_E_z_loss_coeff!(E_z_e_loss, E_z_h_loss, imp0, loss, interface_index, epsR)
    if i < interface_index
        E_z_e_loss[i] = 1.0
        E_z_h_loss[i] = imp0
    else
        E_z_e_loss[i] = (1.0 - loss) / (1.0 + loss)
        E_z_h_loss[i] = imp0 / epsR  / (1.0 + loss)
    end
    return nothing
end

"""
    update_H_y_loss_coeff!()

Update the value of the `Ez` coefficient-loss field depending on the `x` position (with loss layer)

# Arguments
- `H_y_e_loss::Float`: Loss of the `Hy` field due to `E`.
- `H_y_h_loss::Float`: Loss of the `Hy` field due to `H`.
- `imp0::Float` : Impedence of the free space (usually 377.0 Ω)
- `loss::Float` : Loss value introduced after `loss_layer_index`
- `loss_layer_index::Integer` : Index of the `x` position where the loss is introduced
"""
@parallel_indices (i) function update_H_y_loss_coeff!(H_y_e_loss, H_y_h_loss, imp0, loss, loss_layer_index)
    if i < loss_layer_index
        H_y_h_loss[i] = 1.0
        H_y_e_loss[i] = 1.0 / imp0
    else
        H_y_h_loss[i] = (1.0 - loss) / (1.0 + loss)
        H_y_e_loss[i] = 1.0 / imp0 / (1.0 + loss)
    end
    return nothing
end

"""
    update_H_y_loss_coeff!()

Update the value of the `Ez` coefficient-loss field depending on the `x` position (without loss layer)

# Arguments
- `H_y_e_loss::Float`: Loss of the `Hy` field due to `E`.
- `H_y_h_loss::Float`: Loss of the `Hy` field due to `H`.
- `imp0::Float` : Impedence of the free space (usually 377.0 Ω)
"""
@parallel_indices (i) function update_H_y_loss_coeff!(H_y_e_loss, H_y_h_loss, imp0)
        H_y_h_loss[i] = 1.0
        H_y_e_loss[i] = 1.0 / imp0
    return nothing
end

"""
    add_source(it, Cdt_dx, width, delay, location)

Add an additive source as a Gaussian (exponential function) at a specific location. 
"""
function add_source(it, Cdt_dx, width, delay, location)
    return exp(-(it + delay - (-delay) - location / Cdt_dx)^2 / width)
end

"""
    add_source(it, Cdt_dx, width, delay, location)

Add an additive source as a sin function at a specific location. 
"""
function add_source(it, Cdt_dx, location, N_lambda)
    return sin(2.0 * pi / N_lambda * (Cdt_dx * it - location))
end

"""
    ABC_bc(E_z)

Apply Absorbing Boundary Condition to the left of the `Ez` field.
"""
function ABC_bc(E_z)
    E_z[1] = E_z[2]
    return nothing
end

"""
    FDTD_1D(nx_, nt_, nvis_; src="exp", do_visu=false, do_test=false)

Use the Finite Difference Time Domain (FDTD) solver to solve Maxwell's equations

# Arguments
- `nx_::Integer`: Number of x discretization-steps.
- `nt_::Integer`: Number of timesteps.
- `nvis_::Integer`: Number of steps between visualisation output.
- `src::String=exp` : Type of addiditve source (exp or sin).
- `do_visu::Boolean=false`: Perform visualisation.
- `do_test::Boolean=false`: Perform testing (generate a jld reference file).
"""
function FDTD_1D(nx_, nt_, nvis_; src="exp", do_visu=false, do_test=false)
    # Physics
    imp0 = 377.0                        # free space impedance
    if src == "exp"
        loss             = 0.02         # loss factor
        interface_index  = 100          # interface index between free space and dielectric
        epsR             = 9.0          # relative permittivity
        loss_layer_index = 180          # loss layer index
    elseif src == "sin"
        loss             = 0.0253146    # loss factor
        interface_index  = 100          # interface index between free space and dielectric
        epsR             = 4.0          # relative permittivity
        N_lambda         = 40.0         # number of points per wavelengths
    end
    TSFS_boundary_index  = 50          # TSFS boundary index
    
    # Numerics
    nx     = nx_                # number of cells
    nt     = nt_                # number of time steps
    nvis   = nvis_              # visualization interval
    Cdt_dx = 1.0                # Courant number: c * dt/dx

    if src == "exp"
        width    = 100.0        # width of the Gaussian pulse
        location = 30.0         # location of the Gaussian pulse
    elseif src == "sin"
        location = 0.0          # location is 0 for sin source
    end


    # Electric and Magnetic field initialization
    E_z = @zeros(nx + 1)
    H_y = @zeros(nx + 1)

    # Array to store max Ez across all iterations
    max_E_z = @zeros(nx + 1)

    # Lossy coefficient arrays initialization
    E_z_e_loss = @zeros(nx + 1)
    E_z_h_loss = @zeros(nx + 1)
    H_y_e_loss = @zeros(nx + 1)
    H_y_h_loss = @zeros(nx + 1)

    # Update E_z and H_y loss coefficients
    if src == "exp"
        @parallel update_E_z_loss_coeff!(E_z_e_loss, E_z_h_loss, imp0, loss, interface_index, loss_layer_index, epsR)
        @parallel update_H_y_loss_coeff!(H_y_e_loss, H_y_h_loss, imp0, loss, loss_layer_index)
    elseif src == "sin"
        @parallel update_E_z_loss_coeff!(E_z_e_loss, E_z_h_loss, imp0, loss, interface_index, epsR)
        @parallel update_H_y_loss_coeff!(H_y_e_loss, H_y_h_loss, imp0)
    end

    # Visualisation environment for cluster
    if do_visu
        # plotting environment
        ENV["GKSwstype"]="nul"
        if isdir("../docs/viz_out_1D")==false mkdir("../docs/viz_out_1D") end
        loadpath = "../docs/viz_out_1D/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        iframe = 0
    end

    # Time stepping
    for it in 1:nt
        
        # Absorbing boundary conditions on H_y
        # H_y[end] = H_y[end-1]

        # Update magnetic field
        @parallel update_H_y!(H_y, E_z, H_y_e_loss, H_y_h_loss)

        # Correction H_y
        if src == "exp"
            correct_H_y = add_source(it, Cdt_dx, width, 0.0, location)
        elseif src == "sin"
            correct_H_y = add_source(it, Cdt_dx, location, N_lambda)
        end
        H_y[TSFS_boundary_index] -= correct_H_y * H_y_e_loss[TSFS_boundary_index]

        # Absorbing boundary conditions on E_z (only left side)
        ABC_bc(E_z)

        # Update electric field
        @parallel update_E_z!(H_y, E_z, E_z_e_loss, E_z_h_loss)

        # Correction E_z
        if src == "exp"
            correction_E_z = add_source(it, Cdt_dx, width, 0.5, location)
        elseif src == "sin"
            correction_E_z = add_source(it, Cdt_dx, location, N_lambda)
        end
        E_z[TSFS_boundary_index + 1] += correction_E_z

        # Color palette
        colors = ColorSchemes.davos10.colors

        # visualization
        if do_visu && (it % nvis == 0)

            # Loss layer defined only for exp source
            if src=="exp"
                p1 = plot(E_z, label=L"$E_z$", title="\$E_z\$ at it=$it", ylims=(-1.0, 1.0), xlims=(0,nx), legend=:topright, dpi=300, width=2)

                vspan!([0, interface_index], color=colors[4], alpha=0.2, label="")
                vspan!([interface_index, loss_layer_index], color=colors[6], alpha=0.2, label="")
                vspan!([loss_layer_index, nx], color=colors[8], alpha=0.2, label="")
            else
                p1 = plot(E_z, label=L"$E_z$", title="\$E_z\$ at it=$it", ylims=(-2.0, 2.0), xlims=(0,nx), legend=:topright, dpi=300, width=2)

                vspan!([0, interface_index], color=colors[4], alpha=0.2, label="")
                vspan!([interface_index, nx], color=colors[8], alpha=0.2, label="")
            end

            vline!([TSFS_boundary_index], color=colors[3], linestyle=:solid, label="TSFS boundary")
            vline!([interface_index], color=colors[2], linestyle=:dash, label="Interface free space - dielectric")
            if src=="exp"
                vline!([loss_layer_index], color=colors[1], linestyle=:dashdot, label="Lossy layer index")
            end

            png(p1, @sprintf("../docs/viz_out_1D/1D_additive_source_lossy_layer_%04d.png",iframe+=1))

        end

        # Special visualisation for sin 
        if src == "sin"
            for j in 1:nx+1
                if abs(E_z[j]) > max_E_z[j]
                    max_E_z[j] = abs(E_z[j])
                end
            end
        end

    end

    # Special visualisation for sin
    if src == "sin"
        p3 = plot(max_E_z, label=L"$E_z$", title="Maximum \$E_z\$ across all iterations", ylims=(0.0, 1.5), xlims=(0,nx), legend=:topright, dpi=300)
        savefig(p3, "../docs/1D_additive_source_TSFS_max_E_z.png")
    end
    
    # Testing
    if do_test == true
        if USE_GPU
            save("../test/ref_Ez_1D_gpu.jld", "data", E_z)         # store case for reference testing
        else
            save("../test/ref_Ez_1D_cpu.jld", "data", E_z)         # store case for reference testing
        end
    end

    return Array(E_z)
end

FDTD_1D(200, 450, 10; do_visu=true, src="exp")
#FDTD_1D(200, 450, 10; do_visu=true, src="sin")
#FDTD_1D(200, 450, 10; do_visu=false, do_test=true, src="exp")
