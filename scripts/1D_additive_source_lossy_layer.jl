const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 1, inbounds=true)
end
using Plots, LaTeXStrings, Colors, ColorSchemes

plot_font = "Computer Modern"
default(fontfamily=plot_font, framestyle=:box, label=true, grid=true, labelfontsize=11, tickfontsize=11, titlefontsize=13)

@parallel_indices (i) function update_H_y!(H_y, E_z, H_y_e_loss, H_y_h_loss)
    n = length(H_y)
    if i < n
        H_y[i] = H_y_h_loss[i] * H_y[i] + H_y_e_loss[i] * (E_z[i+1] - E_z[i]) 
    end
    return nothing
end

@parallel_indices (i) function update_E_z!(H_y, E_z, E_z_e_loss, E_z_h_loss) 
    if i > 1
         E_z[i] = E_z_e_loss[i] * E_z[i] + E_z_h_loss[i] * (H_y[i] - H_y[i-1])
    end
    return nothing
end

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

function correct_E_z(it, Cdt_dx, width, delay, location)
    return exp(-(it + delay - (-delay) - location / Cdt_dx)^2 / width)
end

function ABC_bc(E_z)
    E_z[1] = E_z[2]
    return nothing
end
    
function FDTD_1D(; do_visu=false)
    # Physics
    imp0 = 377.0            # free space impedance
    loss = 0.02             # loss factor
    loss_layer_index = 180  # loss layer index
    interface_index = 100   # interface index between free space and dielectric
    epsR = 9.0              # relative permittivity
    TSFS_boundary = 50      # TSFS boundary

    # Numerics
    nx = 200                # number of cells
    nt = 450                # number of time steps
    nvis = 10               # visualization interval
    Cdt_dx = 1.0            # Courant number c * dt/dx
    width = 100.0           # width of the Gaussian pulse
    location = 30.0         # location of the Gaussian pulse

    # Electric and Magnetic field initialization
    E_z = @zeros(nx + 1)
    H_y = @zeros(nx + 1)

    # Lossy coefficient arrays initialization
    E_z_e_loss = @zeros(nx + 1)
    E_z_h_loss = @zeros(nx + 1)
    H_y_e_loss = @zeros(nx + 1)
    H_y_h_loss = @zeros(nx + 1)

    # Update E_z and H_y loss coefficients
    @parallel update_E_z_loss_coeff!(E_z_e_loss, E_z_h_loss, imp0, loss, interface_index, loss_layer_index, epsR)
    @parallel update_H_y_loss_coeff!(H_y_e_loss, H_y_h_loss, imp0, loss, loss_layer_index)

    # Time stepping
    for it in 1:nt
        
        # Absorbing boundary conditions on H_y
        #H_y[end] = H_y[end-1]

        # Update magnetic field
        @parallel update_H_y!(H_y, E_z, H_y_e_loss, H_y_h_loss)

        # Correction H_y
        correct_H_y = correct_E_z(it, Cdt_dx, width, 0.0, location)
        H_y[TSFS_boundary] -= correct_H_y * H_y_e_loss[TSFS_boundary]

        # Absorbing boundary conditions on E_z (only left side)
        ABC_bc(E_z)

        # Update electric field
        @parallel update_E_z!(H_y, E_z, E_z_e_loss, E_z_h_loss)

        # Correction E_z
        correction_E_z = correct_E_z(it, Cdt_dx, width, 0.5, location)
        E_z[TSFS_boundary + 1] += correction_E_z

        # Utility to save figures
        save_file = false

        # Color palette
        colors = ColorSchemes.davos10.colors

        # visualization
        if do_visu && (it % nvis == 0)
            p1 = plot(E_z, label=L"$E_z$", title="\$E_z\$ at it=$it", ylims=(-1.0, 1.0), xlims=(0,nx), legend=:topright)
            
            vspan!([0, interface_index], color=colors[4], alpha=0.2, label="")
            vspan!([interface_index, loss_layer_index], color=colors[6], alpha=0.2, label="")
            vspan!([loss_layer_index, nx], color=colors[8], alpha=0.2, label="")
            
            vline!([interface_index], color=colors[2], linestyle=:dash, label="Interface free space - dielectric")
            vline!([loss_layer_index], color=colors[1], linestyle=:dashdot, label="Lossy layer index")
            
            display(p1)

            #p2 = plot(H_y, label=L"$H_y$", title="\$H_y\$ at it=$it", ylims=(-0.05, 0.05))
            #display(p2)

            if (it == 100 || it == 140 || it == 190) && save_file == true
                savefig(p1, "1D_additive_source_TSFS_$it.png")
            end

            sleep(0.5)
        end

    end

    return
end

FDTD_1D(do_visu=true)

