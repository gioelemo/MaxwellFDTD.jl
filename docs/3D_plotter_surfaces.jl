using PlotlyJS,  LaTeXStrings

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


    xc, yc, zc = LinRange(0, lx, nx_pml), LinRange(0, ly, ny_pml), LinRange(0, lz, nz_pml)

    # ----- Plot Ex field at nz/2 -----
    layout1 = Layout(
        title="Ex field at nz/2",
        scene=attr(
            xaxis=attr(title="Lx"),
            yaxis=attr(title="Ly"),
            zaxis=attr(title="Ex[:, :, nz/2]"),
            camera=attr(eye=attr(x=1.5, y=1.5, z=1.5))
        )
    )

    z_data1 = Ex[:, :, Int(nz / 2)]
    p1 = plot(surface(z=z_data1, x=xc, y=yc, showscale=true, aspectratio=attr(x=1, y=1, z=1)), layout1)
    savefig(p1, "Ex_field_3D.png", width=600, height=500)

    # ----- Plot Ey field at nz/2 -----
    layout2 = Layout(
        title="Ey field at nz/2",
        scene=attr(
            xaxis=attr(title="Lx"),
            yaxis=attr(title="Ly"),
            zaxis=attr(title="Ey[:, :, nz/2]"),
            camera=attr(eye=attr(x=1.5, y=1.5, z=1.5))
        )
    )

    z_data2 = Ey[:, :, Int(nz / 2)]
    p2 = plot(surface(z=z_data2, x=xc, y=yc, showscale=true, aspectratio=attr(x=1, y=1, z=1)), layout2)
    savefig(p2, "Ey_field_3D.png", width=600, height=500)

    # ----- Plot Ez field at nz/2 ------
    layout3 = Layout(
        title="Ez field at nz/2",
        scene=attr(
            xaxis=attr(title="Lx"),
            yaxis=attr(title="Ly"),
            zaxis=attr(title="Ez[:, :, nz/2]"),
            camera=attr(eye=attr(x=1.5, y=1.5, z=1.5))
        )
    )

    z_data3 = Ez[:, :, Int(nz / 2)]
    p3 = plot(surface(z=z_data3, x=xc, y=yc, showscale=true, aspectratio=attr(x=1, y=1, z=1)), layout3)
    savefig(p3, "Ez_field_3D.png", width=600, height=500)

    # ----- Plot Hx field at nz/2 -----
    layout4 = Layout(
        title="Hx field at nz/2",
        scene=attr(
            xaxis=attr(title="Lx"),
            yaxis=attr(title="Ly"),
            zaxis=attr(title="Hz[:, :, nz/2]"),
            camera=attr(eye=attr(x=1.5, y=1.5, z=1.5))
        )
    )

    z_data4 = Hx[:, :, Int(nz / 2)]
    p4 = plot(surface(z=z_data4, x=xc, y=yc, showscale=true, aspectratio=attr(x=1, y=1, z=1)), layout4)
    savefig(p4, "Hx_field_3D.png", width=600, height=500)

    # ----- Plot Hy field at nz/2 -----
    layout5 = Layout(
        title="Hy field at nz/2",
        scene=attr(
            xaxis=attr(title="Lx"),
            yaxis=attr(title="Ly"),
            zaxis=attr(title="Hy[:, :, nz/2]"),
            camera=attr(eye=attr(x=1.5, y=1.5, z=1.5))
        )
    )

    z_data5 = Hy[:, :, Int(nz / 2)]
    p5 = plot(surface(z=z_data5, x=xc, y=yc, showscale=true, aspectratio=attr(x=1, y=1, z=1)), layout5)
    savefig(p5, "Hy_field_3D.png", width=600, height=500)

    # ----- Plot Hz field at nz/2-1 -----
    layout6 = Layout(
        title="Hz field at nz/2-1",
        scene=attr(
            xaxis=attr(title="Lx"),
            yaxis=attr(title="Ly"),
            zaxis=attr(title="Hy[:, :, nz/2-1]"),
            camera=attr(eye=attr(x=1.5, y=1.5, z=1.5))
        )
    )

    z_data6 = Hz[:, :, Int(nz / 2) - 1]
    p6 = plot(surface(z=z_data6, x=xc, y=yc, showscale=true, aspectratio=attr(x=1, y=1, z=1)), layout6)
    savefig(p6, "Hz_field_3D.png", width=600, height=500)

end

visualise()