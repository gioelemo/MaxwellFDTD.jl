using GLMakie

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

function visualise()
    lx, ly, lz = 40.0, 40.0, 40.0
    nx = 100
    ny = 100
    nz = 100

    # PML parameters
    pml_width = 5
      
    # Extend the grid
    nx_pml, ny_pml , nz_pml = nx + 2 * pml_width, ny + 2 * pml_width, nz + 2 * pml_width

    Ex = zeros(Float32, nx_pml, ny_pml+1, nz_pml+1)
    Ey = zeros(Float32, nx_pml+1, ny_pml, nz_pml+1)
    Ez = zeros(Float32, nx_pml+1, ny_pml+1, nz_pml)

    Hx = zeros(Float32, nx_pml, ny_pml, nz_pml)
    Hy = zeros(Float32, nx_pml, ny_pml, nz_pml)
    Hz = zeros(Float32, nx_pml, ny_pml, nz_pml)

    
    load_array("out_Ex", Ex)
    load_array("out_Ey", Ey)
    load_array("out_Ez", Ez)

    load_array("out_Hx", Hx)
    load_array("out_Hy", Hy)
    load_array("out_Hz", Hz)

    xc, yc, zc = LinRange(0, lx, nx_pml), LinRange(0, ly, ny_pml), LinRange(0, lz, nz_pml)
    #dx, dy, dz = lx / nx, ly / ny, lz/ nz

    #xc = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx)
    #yc = LinRange(-ly / 2 + dy / 2, ly / 2 - dy / 2, ny)
    #zc = LinRange(-lz / 2 + dz / 2, lz / 2 - dz / 2, nz)
    fig = Figure(resolution=(1600, 1000), fontsize=24)
    
    ax1 = Axis3(fig[1, 1]; aspect=(1, 1, 1), title="Ex", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_Ex = contour!(ax1, xc, yc, zc, Ex; alpha=0.05, colormap=:turbo)
    
    ax2 = Axis3(fig[1, 2]; aspect=(1, 1, 1), title="Ey", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_Ey = contour!(ax2, xc, yc, zc, Ey; alpha=0.05, colormap=:turbo)
    
    ax3 = Axis3(fig[1, 3]; aspect=(1, 1, 1), title="Ez", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_Ez = contour!(ax3, xc, yc, zc, Ez; alpha=0.05, colormap=:turbo)

    fig2 = Figure(resolution=(1600, 1000), fontsize=24)
    
    ax11 = Axis3(fig2[1, 1]; aspect=(1, 1, 1), title="Hx", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_Hx = contour!(ax11, xc, yc, zc, Hx; alpha=0.05, colormap=:turbo)
    
    ax22 = Axis3(fig2[1, 2]; aspect=(1, 1, 1), title="Hy", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_Hy = contour!(ax22, xc, yc, zc, Hy; alpha=0.05, colormap=:turbo)
    
    ax33 = Axis3(fig2[1, 3]; aspect=(1, 1, 1), title="Hz", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_Hz = contour!(ax33, xc, yc, zc, Hz; alpha=0.05, colormap=:turbo)
    
    save("3D_plot_E.png", fig)
    save("3D_plot_H.png", fig2)
end

visualise()