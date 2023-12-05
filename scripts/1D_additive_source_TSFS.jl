const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 1, inbounds=true)
end
using Plots

plot_font = "Computer Modern"
default(fontfamily=plot_font, framestyle=:box, label=true, grid=true, labelfontsize=11, tickfontsize=11, titlefontsize=13)

@parallel_indices (i) function update_H_y!(H_y, E_z, imp0)
    # for i in 1:nx-1
    #     H_y[i] = H_y[i] + (E_z[i+1] - E_z[i]) / imp0
    # end
    
    n = length(H_y)
    if i < n
        H_y[i] = H_y[i] + (E_z[i+1] - E_z[i]) / imp0
    end
    
    return nothing
end

@parallel_indices (i) function update_E_z!(H_y, E_z, epsR, imp0)
    # for i in 2:nx-1
    # E_z[i] = E_z[i] + (H_y[i] - H_y[i-1]) * imp0/epsR[i]   
    # end

    n = length(H_y)
    if i > 1
         E_z[i] = E_z[i] + (H_y[i] - H_y[i-1]) * imp0 / epsR[i]
    end

    return nothing
end


function FDTD_1D(; do_visu=false)
    # Physics
    imp0 = 377.0

    # numerics
    nx = 200
    nt = 450
    nvis = 10

    # initialize fields
    E_z = @zeros(nx + 1)
    #H_y = @zeros(nx + 1)
    H_y = @zeros(nx + 1)
    epsR = @zeros(nx + 1)

    # add inomogeneous medium
    epsR = [i < 100 ? 1.0 : 9.0 for i in 1:nx+1]


    # time stepping
    for it in 1:nt
        
        # absorbing boundary conditions on H_y
        #H_y[end] = H_y[end-1]

        # update magnetic field
        @parallel update_H_y!(H_y, E_z, imp0)

        # correcting H_y
        H_y[49] = H_y[49] - exp(-(it - 30.0)^2 / 100.0) / imp0 

        # absorbing boundary conditions on E_z
        E_z[1] = E_z[2]
        E_z[end] = E_z[end-1]

        # update electric field
        @parallel update_E_z!(H_y, E_z, epsR, imp0)

        # correction E_z
        E_z[50] = E_z[50] + exp(- (it + 0.5 - (-0.5) - 30.0)^2 / 100.0)

        save_file = false

        # visualization
        if do_visu && (it % nvis == 0)
            p1 = plot(E_z, label="E_z", title="E_z at t=$it", ylims=(-1.0, 1.0))
            # p2 = plot(H_y, label="H_y")
            # plot!(H_y, label="H_y")
            display(p1)

            if (it == 100 || it == 140) && save_file == true
                savefig(p1, "1D_additive_source_TSFS_$it.png")
            end

            sleep(0.5)
        end

    end


    
    return
end

FDTD_1D(do_visu=true)

