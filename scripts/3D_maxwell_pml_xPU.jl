const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
end

using Printf, Plots, JLD
plot_font = "Computer Modern"
default(fontfamily=plot_font, framestyle=:box, label=true, grid=true, labelfontsize=11, tickfontsize=11, titlefontsize=13)


"""
    save_array(Aname,A)

Store an array in a binary format (used then for plotting)
"""
function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end

@views avx_z(A) = 0.5 .* (A[:, 2:end-1, 1:end-1] .+ A[:, 2:end-1, 2:end])
@views avx_y(A) = 0.5 .* (A[:, 1:end-1, 2:end-1] .+ A[:, 2:end, 2:end-1])

@views avy_x(A) = 0.5 .* (A[1:end-1, :, 2:end-1] .+ A[2:end, :, 2:end-1])
@views avy_z(A) = 0.5 .* (A[2:end-1, :, 1:end-1] .+ A[2:end-1, :, 2:end])

@views avz_x(A) = 0.5 .* (A[1:end-1, 2:end-1, :] .+ A[2:end, 2:end-1, :])
@views avz_y(A) = 0.5 .* (A[2:end-1, 1:end-1, :] .+ A[2:end-1, 2:end, :])

@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

function update_Ex!(Ex, Hy, Hz, σ, ε0, dt, dy, dz)
    #Ex[:, 2:end-1] .+= dt / ε0 .* (-σ .* Ex[:, 2:end-1] .+ diff(Hz, dims=2)./ dy)
    #@inn_y(Ex) = @inn_y(Ex) .+ dt / ε0 .* (-σ .* @inn_y(Ex) .+ @d_ya(Hz) ./ dy)
    
    # println("Ex[:, 2:end-1, 2:end-1]", size(Ex[:, 2:end-1, 2:end-1]))
    # println("diff(Hz, dims=2)", size(diff(Hz, dims=2)))
    # println("diff(Hy, dims=3)", size(diff(Hy, dims=3)))


    #Ex[:, 2:end-1, 2:end-1] .+= dt / ε0 .* (-σ .* Ex[:, 2:end-1, 2:end-1] .+ diff(Hz, dims=2)./ dy .- diff(Hy, dims=3)./dz)

    Ex[:, 2:end-1, 2:end]   .+= dt / ε0 .* (-σ .* Ex[:, 2:end-1, 2:end] .+ diff(Hz, dims=2)./ dy )
    Ex[:, 2:end-1, 1:end-1] .+= dt / ε0 .* (-σ .* Ex[:, 2:end-1, 1:end-1] .+ diff(Hz, dims=2)./ dy )

    Ex[:, 2:end-1, 1:end-1 ]  = avx_z(Ex)

    Ex[:, 2:end, 2:end-1]   .+= dt / ε0 .* (-σ .* Ex[:, 2:end, 2:end-1] .- diff(Hy, dims=3)./ dz )
    Ex[:, 1:end-1, 2:end-1] .+= dt / ε0 .* (-σ .* Ex[:, 1:end-1, 2:end-1] .- diff(Hy, dims=3)./ dz )
    
    Ex[:, 1:end-1, 2:end-1]   = avx_y(Ex)


    return nothing
end

function update_Ey!(Ey, Hx, Hz, σ, ε0, dt, dx, dz)
    #Ey[2:end-1, :] .+= dt / ε0 .* (-σ .* Ey[2:end-1, :] .- diff(Hz, dims=1) ./ dx)
    #@inn_x(Ey) = @inn_x(Ey) .+ dt / ε0 .* (-σ .* @inn_x(Ey) .- @d_xa(Hz) ./ dx)

    #Ey[2:end-1, :, 2:end-1] .+= dt / ε0 .* (-σ .* Ey[2:end-1, :, 2:end-1] .+ diff(Hx, dims=3)./dz .- diff(Hz, dims=1) ./ dx)

    Ey[2:end, :, 2:end-1]   .+= dt / ε0 .* (-σ .* Ey[2:end, :, 2:end-1] .+ diff(Hx, dims=3)./dz)
    Ey[1:end-1, :, 2:end-1] .+= dt / ε0 .* (-σ .* Ey[1:end-1, :, 2:end-1] .+ diff(Hx, dims=3)./dz)

    Ey[1:end-1, :, 2:end-1]   = avy_x(Ey)

    Ey[2:end-1, :, 2:end]   .+= dt / ε0 .* (-σ .* Ey[2:end-1, :, 2:end] .- diff(Hz, dims=1)./dx)
    Ey[2:end-1, :, 1:end-1] .+= dt / ε0 .* (-σ .* Ey[2:end-1, :, 1:end-1] .- diff(Hz, dims=1)./dx)

    Ey[2:end-1, :, 1:end-1]   = avy_z(Ey)

    return nothing
end

function update_Ez!(Ez, Hx, Hy, σ, ε0, dt, dx, dy)


    #Ez[2:end-1, 2:end-1, :] .+= dt / ε0 .* (-σ .* Ez[2:end-1, 2:end-1, :] .+ diff(Hy, dims=1)./dx .- diff(Hx, dims=2) ./ dy)
    
    Ez[2:end-1, 2:end, :]   .+= dt / ε0 .* (-σ .* Ez[2:end-1, 2:end, :] .+ diff(Hy, dims=1)./dx)
    Ez[2:end-1, 1:end-1, :] .+= dt / ε0 .* (-σ .* Ez[2:end-1, 1:end-1, :] .+ diff(Hy, dims=1)./dx)

    Ez[2:end-1, 1:end-1, :]   = avz_y(Ez)

    
    Ez[2:end, 2:end-1, :]   .+= dt / ε0 .* (-σ .* Ez[2:end, 2:end-1, :] .- diff(Hx, dims=2)./dy)
    Ez[1:end-1, 2:end-1, :] .+= dt / ε0 .* (-σ .* Ez[1:end-1, 2:end-1, :] .- diff(Hx, dims=2)./dy)

    Ez[1:end-1, 2:end-1, :]   = avz_x(Ez)


    return nothing
end

# @parallel_indices (i,j) function update_PML_x!(pml_width, pml_alpha, Ex)
#     # for i in 1:pml_width
#     #     Ex[i, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[i, :]
#     #     Ex[end - i + 1, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[end - i + 1, :]
#     #     Ey[:, i] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, i]
#     #     Ey[:, end - i + 1] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, end - i + 1]
#     # end
#     Ex[i, j] = exp(-(pml_width - i) * pml_alpha) * Ex[i, j]
#     Ex[end - i + 1, j] = exp(-(pml_width - i) * pml_alpha) * Ex[end - i + 1, j]

#     return nothing
# end

# @parallel_indices (i,j) function update_PML_y!(pml_width, pml_alpha, Ey)
#     # for i in 1:pml_width
#     #     Ex[i, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[i, :]
#     #     Ex[end - i + 1, :] .= exp(-(pml_width - i) * pml_alpha) .* Ex[end - i + 1, :]
#     #     Ey[:, i] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, i]
#     #     Ey[:, end - i + 1] .= exp(-(pml_width - i) * pml_alpha) .* Ey[:, end - i + 1]
#     # end

#     Ey[j, i] = exp(-(pml_width - i) * pml_alpha) * Ey[j, i]
#     Ey[j, end - i + 1] = exp(-(pml_width - i) * pml_alpha) * Ey[j, end - i + 1]

#     return nothing
# end

function update_Hx!(Hx, Ey, Ez, σ, μ0, dt, dy, dz)
    #Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2) ./ dy .- diff(Ey, dims=1) ./ dx)
    #@all(Hz) = @all(Hz) .+ dt / μ0 .* (-σ .* @all(Hz) .+ @d_ya(Ex) ./ dy .- @d_xa(Ey) ./ dx)

    # println("Hx", size(Hx))
    # println("diff(Ey, dims=3)", size(diff(Ey, dims=3)))
    # println("diff(Ez, dims=2)", size(diff(Ez, dims=2)))

    #Hx .+= dt / μ0 .* (-σ .* Hx .+ diff(Ey, dims=3) ./ dz .- diff(Ez, dims=2) ./ dy)

    Hx .+= dt / μ0 .* (-σ .* Hx .+ diff(Ey, dims=3)[1:end-1,:,:] ./ dz .- diff(Ez, dims=2)[1:end-1,:,:] ./ dy)
    Hx .+= dt / μ0 .* (-σ .* Hx .+ diff(Ey, dims=3)[2:end,:,:] ./ dz .- diff(Ez, dims=2)[2:end,:,:] ./ dy)

    Hx = avx(Hx)

    return nothing
end

function update_Hy!(Hy, Ex, Ez, σ, μ0, dt, dx, dz)
    #Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2) ./ dy .- diff(Ey, dims=1) ./ dx)
    #@all(Hz) = @all(Hz) .+ dt / μ0 .* (-σ .* @all(Hz) .+ @d_ya(Ex) ./ dy .- @d_xa(Ey) ./ dx)

    # println("Hy", size(Hy))
    # println("diff(Ez, dims=1)", size(diff(Ez, dims=1)))
    # println("diff(Ex, dims=3)", size(diff(Ex, dims=3)))

    #Hy .+= dt / μ0 .* (-σ .* Hy .+ diff(Ez, dims=1) ./ dx .- diff(Ex, dims=3) ./ dz)

    Hy .+= dt / μ0 .* (-σ .* Hy .+ diff(Ez, dims=1)[:,1:end-1,:] ./ dx .- diff(Ex, dims=3)[:,1:end-1,:] ./ dz)
    Hy .+= dt / μ0 .* (-σ .* Hy .+ diff(Ez, dims=1)[:,2:end,:] ./ dx .- diff(Ex, dims=3)[:,2:end,:] ./ dz)

    Hy = avy(Hy)

    return nothing
end


function update_Hz!(Hz, Ex, Ey, σ, μ0, dt, dy, dx)
    #Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2) ./ dy .- diff(Ey, dims=1) ./ dx)
    #@all(Hz) = @all(Hz) .+ dt / μ0 .* (-σ .* @all(Hz) .+ @d_ya(Ex) ./ dy .- @d_xa(Ey) ./ dx)

    # println("Hz", size(Hz))
    # println("diff(Ex, dims=2)", size(diff(Ex, dims=2)))
    # println("diff(Ey, dims=1)", size(diff(Ey, dims=1)))

    #Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2) ./ dy .- diff(Ey, dims=1) ./ dx)

    Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2)[:,:,1:end-1] ./ dy .- diff(Ey, dims=1)[:,:,1:end-1] ./ dx)
    Hz .+= dt / μ0 .* (-σ .* Hz .+ diff(Ex, dims=2)[:,:,2:end] ./ dy .- diff(Ey, dims=1)[:,:,2:end] ./ dx)

    Hz = avz(Hz)
    return nothing
end
    
@views function maxwell(nz_, nt_, nvis_, pml_alpha_; do_visu=false, do_check=true, do_test=true)
    # physics
    lx, ly, lz = 40.0, 40.0, 40.0
    ε0 = 1.0
    μ0 = 1.0
    σ = 1.0
    
    # numerics
    nx, ny, nz = nz_ , nz_ , nz_

    # PML parameters
    pml_width = 0
    pml_alpha = pml_alpha_
     
    # Extend the grid
    nx_pml, ny_pml , nz_pml = nx + 2 * pml_width, ny + 2 * pml_width, nz + 2 * pml_width

    # Other numerics parameters
    dx, dy, dz = lx / nx_pml, ly / ny_pml, lz/ nz_pml
    xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx_pml)
    yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny_pml)
    zc = LinRange(-lz / 2 + dz / 2, lz / 2 - dz / 2, nz_pml)

    println("numerics okay")

    dt = min(dx, dy, dz)^2 / (1 / ε0 / μ0) / 4.1
    nt = nt_
    nout = nvis_

    # initial conditions
    Ex = @zeros(nx_pml, ny_pml + 1, nz_pml + 1)
    Ey = @zeros(nx_pml + 1, ny_pml, nz_pml + 1)
    Ez = @zeros(nx_pml + 1, ny_pml + 1, nz_pml)
    Hx = @zeros(nx_pml, ny_pml, nz_pml)
    Hy = @zeros(nx_pml, ny_pml, nz_pml)
    Hz = @zeros(nx_pml, ny_pml, nz_pml)


    Hx = Data.Array([exp(-(xc[ix] - lx / 2)^2 - (yc[iy] - ly / 2)^2 - (zc[iz] - lz / 2)^2) for ix = 1:nx_pml, iy = 1:ny_pml, iz = 1:nz_pml])
    #Hx = Data.Array([exp(-(xc[ix])^2 - (yc[iy])^2 - (zc[iz])^2) for ix = 1:nx_pml, iy = 1:ny_pml, iz = 1:nz_pml])

    
    Hy = copy(Hx)
    Hz = copy(Hx)


    println("initial conditions okay")


    # visualisation for cluster
    if do_visu
        # plotting environment
        ENV["GKSwstype"]="nul"
        if isdir("../docs/viz_out_3D")==false mkdir("../docs/viz_out_3D") end
        loadpath = "../docs/viz_out_3D/"; anim = Animation(loadpath,String[])
        println("Animation directory: $(anim.dir)")
        iframe = 0
    end



    # timestepping
    for it in 1:nt
        # Update E
        update_Ex!(Ex, Hy, Hz, σ, ε0, dt, dy, dz)
        #println("update ex okay")
        update_Ey!(Ey, Hx, Hz, σ, ε0, dt, dx, dz)
        #println("update ey okay")
        update_Ez!(Ez, Hx, Hy, σ, ε0, dt, dx, dy)
        #println("update ez okay")

        # Update PML
        #@parallel (1:pml_width, 1:size(Ex, 2)) update_PML_x!(pml_width, pml_alpha, Ex)
        #@parallel (1:pml_width, 1:size(Ey, 1)) update_PML_y!(pml_width, pml_alpha, Ey)

        # Update H
        update_Hx!(Hx, Ey, Ez, σ, μ0, dt, dy, dz)
        #println("update hx okay")
        update_Hy!(Hy, Ex, Ez, σ, μ0, dt, dx, dz)
        #println("update hy okay")
        update_Hz!(Hz, Ex, Ey, σ, μ0, dt, dy, dx)
        #println("update hz okay")
        
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
        println(it)
    end

    # Save array for visualization

    save_array("../docs/out_Ex",convert.(Float32,Array(Ex)))
    save_array("../docs/out_Ey",convert.(Float32,Array(Ey)))
    save_array("../docs/out_Ez",convert.(Float32,Array(Ez)))

    save_array("../docs/out_Hx",convert.(Float32,Array(Hx)))
    save_array("../docs/out_Hy",convert.(Float32,Array(Hy)))
    save_array("../docs/out_Hz",convert.(Float32,Array(Hz)))

    # testing
    if do_test == true
        if USE_GPU
            save("../test/ref_Hz_3D_gpu.jld", "data", Hz)         # store case for reference testing
        else
            save("../test/ref_Hz_3D_cpu.jld", "data", Hz)         # store case for reference testing
        end
    end

    return Array(Ez)
end

# ny, nt, nvis
#maxwell(101, 1000, 100; do_visu=false, do_test=true)

#maxwell(50, 10, 10, 0.25; do_visu=false, do_test=true)

#maxwell(256, 15000, 100, 0.0; do_visu=true, do_test=false)
#maxwell(256, 15000, 100, 5.0; do_visu=true, do_test=false)
#maxwell(256, 15000, 100, 0.1; do_visu=true, do_test=false)


maxwell(100,100,100,0.0; do_visu=false, do_test=false)