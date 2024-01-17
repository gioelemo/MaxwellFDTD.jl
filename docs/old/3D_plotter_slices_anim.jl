using Plots

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

function visualise()

    plot_font = "Computer Modern"
    default(fontfamily=plot_font, framestyle=:box, label=true, grid=true, labelfontsize=11, tickfontsize=11, titlefontsize=13)
    lx, ly, lz = 40.0, 40.0, 40.0
    nx = 100
    ny = 100
    nz = 100

    # PML parameters
    pml_width = 0
      
    # Extend the grid
    nx_pml, ny_pml , nz_pml = nx + 2 * pml_width, ny + 2 * pml_width, nz + 2 * pml_width

    Ex = zeros(Float32, nx_pml, ny_pml+1, nz_pml+1)
    Ey = zeros(Float32, nx_pml+1, ny_pml, nz_pml+1)
    Ez = zeros(Float32, nx_pml+1, ny_pml+1, nz_pml)

    Hx = zeros(Float32, nx_pml-1, ny_pml, nz_pml)
    Hy = zeros(Float32, nx_pml, ny_pml-1, nz_pml)
    Hz = zeros(Float32, nx_pml, ny_pml, nz_pml-1)
 
    load_array("out_Ex", Ex)
    load_array("out_Ey", Ey)
    load_array("out_Ez", Ez)

    load_array("out_Hx", Hx)
    load_array("out_Hy", Hy)
    load_array("out_Hz", Hz)

    # Animate Ex field slices
    anim1 = @animate for iz in 1:nz
        heatmap(Ex[:, :, iz]', color=:turbo, aspect_ratio=:equal, xlims=(0, nx), ylims=(0, ny), title="Ex Slice $iz")
    end
    gif(anim1, "Ex_slice.gif", fps = 15)

    # Animate Ey field slices
    anim2 = @animate for iz in 1:nz
        heatmap(Ey[:, :, iz]', color=:turbo, aspect_ratio=:equal, xlims=(0, nx), ylims=(0, ny), title="Ey Slice $iz")
    end
    gif(anim2, "Ey_slice.gif", fps = 15)

    # Animate Ez field slices
    anim3 = @animate for iy in 1:ny
        heatmap(Ez[:, iy, :]', color=:turbo, aspect_ratio=:equal, xlims=(0, nx), ylims=(0, nz), title="Ez Slice $iy")
    end
    gif(anim3, "Ez_slice.gif", fps = 15)

    # Animate Hx field slices
    anim4 = @animate for iz in 1:nz
        heatmap(Hx[:, :, iz]', color=:turbo, aspect_ratio=:equal, xlims=(0, nx - 1), ylims=(0, ny), title="Hx Slice $iz")  
    end
    gif(anim4, "Hx_slice.gif", fps = 15)

    # Animate Hy field slices
    anim5 = @animate for iz in 1:nz
        heatmap(Hy[:, :, iz]', color=:turbo, aspect_ratio=:equal, xlims=(0, nx), ylims=(0, ny - 1),  title="Hy Slice $iz")
    end
    gif(anim5, "Hy_slice.gif", fps = 15)

    # Animate Hz field slices
    anim6 = @animate for iy in 1:ny
        heatmap(Hz[:, iy, :]', color=:turbo, aspect_ratio=:equal, xlims=(0, nx), ylims=(0, nz - 1), title="Hz Slice $iy")
    end
    gif(anim6, "Hz_slice.gif", fps = 15)
end

visualise()