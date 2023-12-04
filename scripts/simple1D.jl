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

function FDTD_1D(; do_visu=false)
    # Physics
    imp0 = 377.0

    # numerics
    nx = 200
    nt = 1000
    nvis = 50

    # init

    E_z = @zeros(nx)
    H_y = @zeros(nx)

    E_z_values = @zeros(nt)

    # time stepping
    for it in 1:nt

        # update magnetic field
        for i in 1:nx-1
            H_y[i] = H_y[i] + (E_z[i+1] - E_z[i]) / imp0
        end

        # update electric field
        for i in 2:nx-1
            E_z[i] = E_z[i] + (H_y[i] - H_y[i-1]) * imp0
        end

        # source
        E_z[1] = exp(-(it-30.0)^2 / 100.0)

        if do_visu && (it % nvis == 0)
            p1 = plot(E_z, label="E_z", title="E_z at t=$it")
            p2 = plot(H_y, label="H_y")
            display(p1)
            sleep(0.5)
        end

        E_z_values[it] = E_z[50]
        
    end

    #display(plot(E_z_values, label="E_z"))

    return
end

FDTD_1D(do_visu=true)