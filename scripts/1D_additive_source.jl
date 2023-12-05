const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 1, inbounds=true)
end
using Plots

include("plotting.jl")

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

@parallel_indices (i) function update_E_z!(H_y, E_z, imp0)
    # for i in 2:nx-1
    # E_z[i] = E_z[i] + (H_y[i] - H_y[i-1]) * imp0   
    # end

    n = length(H_y)
    if i > 1
         E_z[i] = E_z[i] + (H_y[i] - H_y[i-1]) * imp0
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
    H_y = @zeros(nx + 1)

    # time stepping
    for it in 1:nt
        
        # absorbing boundary conditions on H_y
        H_y[end] = H_y[end-1]

        # update magnetic field
        @parallel update_H_y!(H_y, E_z, imp0)

        # absorbing boundary conditions on E_z
        E_z[1] = E_z[2]

        # update electric field
        @parallel update_E_z!(H_y, E_z, imp0)

        # point source
        E_z[51] = exp(-(it-30.0)^2 / 100.0)

        # visualization
        if do_visu && (it % nvis == 0)
            p1 = plot(E_z, label="E_z", title="E_z at t=$it", ylims=(-1.0, 1.0))
            # p2 = plot(H_y, label="H_y")
            display(p1)

            sleep(0.5)
        end

    end


    
    return
end

FDTD_1D(do_visu=true)

