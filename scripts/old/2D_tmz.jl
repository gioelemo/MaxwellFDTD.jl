const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=true)
end
using Plots, LaTeXStrings, Colors, ColorSchemes

plot_font = "Computer Modern"
default(fontfamily=plot_font, framestyle=:box, label=true, grid=true, labelfontsize=11, tickfontsize=11, titlefontsize=13)


function update_H_for!(H_y, H_x, E_z, H_y_e_loss, H_y_h_loss, H_x_e_loss, H_x_h_loss)
    nx = size(H_y, 1) - 1
    ny = size(H_y, 2) - 1
    #print(H_x[nx, ny])

    #savefig(heatmap(H_x, title="H_x", dpi=300), "test.png")
    

    for i in 1:nx+1
        for j in 1:ny
            H_x[i, j] = H_x_h_loss[i, j] * H_x[i, j] - H_x_e_loss[i, j] * (E_z[i, j + 1] - E_z[i, j])
            
        end
    end

    for i in 1:nx
        for j in 1:ny+1
            H_y[i, j] = H_y_h_loss[i, j] * H_y[i, j] + H_y_e_loss[i, j] * (E_z[i + 1, j] - E_z[i, j])
        end
    end
end


function  update_E_z_for!(H_y, H_x, E_z, E_z_e_loss, E_z_h_loss)

    nx = size(H_y, 1) - 1
    ny = size(H_y, 2) - 1 
    print(nx)

    for i in 2:nx
        for j in 2:ny
            E_z[i, j] = E_z_e_loss[i, j] * E_z[i, j] 
            + E_z_h_loss[i, j] * ((H_y[i, j] - H_y[i-1, j]) - (H_x[i, j] - H_x[i, j-1]))
        end
    end

end

@parallel_indices (i, j) function update_E_z_loss_coeff!(E_z_e_loss, E_z_h_loss, imp0, Cdt_dx)
        E_z_e_loss[i, j] = 1.0
        E_z_h_loss[i, j] = imp0 * Cdt_dx
    return nothing
end

@parallel_indices (i, j) function update_H_y_loss_coeff!(H_y_e_loss, H_y_h_loss, imp0, Cdt_dx)
        H_y_h_loss[i, j] = 1.0
        H_y_e_loss[i, j] = Cdt_dx / imp0
    return nothing
end

@parallel_indices (i, j) function update_H_x_loss_coeff!(H_x_e_loss, H_x_h_loss, imp0, Cdt_dx)
        H_x_h_loss[i, j] = 1.0
        H_x_e_loss[i, j] = Cdt_dx / imp0
    return nothing
end


function correct_E_z(it, Cdt_dx, ppw, location)
    arg = pi * ((Cdt_dx * it - location) / ppw - 1.0)
    println("it= ", it, " arg = ", arg)
    arg = arg * arg

    return (1.0 - 2.0 * arg) * exp(-arg)
end


function FDTD_2D(; bc="exp", do_visu=false)
    # Physics
    imp0 = 377.0                # free space impedance
    ppw = 20
    

    # Numerics
    nx     = 101                # number of cells
    ny     = 81

    nt     = 300                # number of time steps
    nvis   = 10                 # visualization interval
    Cdt_dx = 1.0 /sqrt(2.0)     # Courant number: c * dt/dx

    location = 0.0

    # Electric and Magnetic field initialization
    H_x = @zeros(nx + 1, ny )
    H_y = @zeros(nx, ny + 1)
    E_z = @zeros(nx + 1, ny + 1)
    

    # Lossy coefficient arrays initialization
    E_z_e_loss = @zeros(nx + 1, ny + 1)
    E_z_h_loss = @zeros(nx + 1, ny + 1)
    H_y_e_loss = @zeros(nx , ny + 1)
    H_y_h_loss = @zeros(nx , ny + 1)
    H_x_e_loss = @zeros(nx + 1, ny )
    H_x_h_loss = @zeros(nx + 1, ny )

    # Update E_z and H_y loss coefficients

    @parallel update_E_z_loss_coeff!(E_z_e_loss, E_z_h_loss, imp0, Cdt_dx)
    @parallel update_H_y_loss_coeff!(H_y_e_loss, H_y_h_loss, imp0, Cdt_dx)
    @parallel update_H_x_loss_coeff!(H_x_e_loss, H_x_h_loss, imp0, Cdt_dx)

    println(E_z_e_loss[nx ÷ 2, ny ÷ 2], " ", E_z_h_loss[nx ÷ 2, ny ÷ 2], " ", H_y_e_loss[nx ÷ 2, ny ÷ 2],  " ", H_y_h_loss[nx ÷ 2, ny ÷ 2])

    p3 = heatmap(E_z_e_loss, title="E_z_e_loss", dpi=300)
    p4 = heatmap(E_z_h_loss, title="E_z_h_loss", dpi=300)
    p5 = heatmap(H_y_e_loss, title="H_y_e_loss", dpi=300)
    #display(p5)
    
    # Time stepping
    for it in 1:nt
        if it== 1 
            println("start timestepping")
        end
        # Absorbing boundary conditions on H_y
        # H_y[end] = H_y[end-1]

        # Update magnetic field
        #@parallel update_H!(H_y, H_x, E_z, H_y_e_loss, H_y_h_loss, H_x_e_loss, H_x_h_loss)
        update_H_for!(H_y, H_x, E_z, H_y_e_loss, H_y_h_loss, H_x_e_loss, H_x_h_loss)
        println(H_x[nx ÷ 2, ny ÷ 2])
        println(H_y[nx ÷ 2, ny ÷ 2])
        println("update h okay")

        # # Correction H_y
        # if bc == "exp"
        #     correct_H_y = correct_E_z(it, Cdt_dx, width, 0.0, location)
        # elseif bc == "sin"
        #     correct_H_y = correct_E_z2(it, Cdt_dx, location, N_lambda)
        # end

        #correct_H_y = correct_E_z(it, Cdt_dx, ppw, location)
        
        #H_y[nx ÷ 2, ny ÷ 2] -= correct_H_y * H_y_e_loss[nx ÷ 2, ny ÷ 2]

        # Absorbing boundary conditions on E_z (only left side)
        #ABC_bc(E_z)

        # Update electric field
        #@parallel update_E_z!(H_y, H_x, E_z, E_z_e_loss, E_z_h_loss)
        update_E_z_for!(H_y, H_x, E_z, E_z_e_loss, E_z_h_loss)
        println(E_z[nx ÷ 2, ny ÷ 2])
        println("update e okay")

        # Correction E_z
        correction_E_z = correct_E_z(it, Cdt_dx, ppw, location)
        println("correction: ", correction_E_z)
        E_z[nx ÷ 2, ny ÷ 2] = correction_E_z
        println("correction okay")

        # Utility to save figures
        save_file = false

        # Color palette
        colors = ColorSchemes.davos10.colors

        # visualization
        if do_visu && (it % nvis == 0)
            p1 = heatmap(E_z')

            println("Mean of E_z = ", sum(E_z)/(nx*ny))

            display(p1)
        
            if save_file == true
                savefig(p1, "./docs/2D_additive_source_TSFS_$it.png")
            end

            sleep(0.2)
        end

    
    end

    return
end

FDTD_2D(do_visu=true, bc="exp")
