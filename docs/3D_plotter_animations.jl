using Plots

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid = open(fname, "r"); read!(fid, A); close(fid)
end

plot_font = "Computer Modern"
default(fontfamily=plot_font, framestyle=:box, label=true, grid=true, labelfontsize=11, tickfontsize=11, titlefontsize=13)
lx, ly, lz = 40.0, 40.0, 40.0
nx, ny, nz = 256, 256, 100

# PML parameters
pml_width = 10
alpha = "0.0"

# Extend the grid
nx_pml, ny_pml, nz_pml = nx + 2 * pml_width, ny + 2 * pml_width, nz + 2 * pml_width

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

function create_animation(field, field_name)
    animation = @animate for k in 1+pml_width:nz_pml-pml_width
        heatmap(field[:, :, k]', xlabel="nx", ylabel="ny", zlabel=field_name, title="\$$field_name\$ at it=$k",
                color=:turbo, aspect_ratio=:equal, xlims=(1, nx_pml), ylims=(1, ny_pml))
        rect_x = [pml_width, nx_pml-pml_width+1, nx_pml-pml_width+1, pml_width, pml_width]
        rect_y = [pml_width, pml_width, ny_pml-pml_width+1, nx_pml-pml_width+1, pml_width]
        plot!(rect_x, rect_y, line=:black, linewidth=2, fillalpha=0, legend=false)
    end
    return animation
end

Ex_animation = create_animation(Ex, "E_x")
Ey_animation = create_animation(Ey, "E_y")
Ez_animation = create_animation(Ez, "E_z")
Hx_animation = create_animation(Hx, "H_x")
Hy_animation = create_animation(Hy, "H_y")
Hz_animation = create_animation(Hz, "H_z")

gif(Ex_animation, "Ex_3D_pml_nx_$(nx)_ny_$(ny)_nz_$(nz)_alpha_$alpha.gif", fps=10)
gif(Ey_animation, "Ey_3D_pml_nx_$(nx)_ny_$(ny)_nz_$(nz)_alpha_$alpha.gif", fps=10)
gif(Ez_animation, "Ez_3D_pml_nx_$(nx)_ny_$(ny)_nz_$(nz)_alpha_$alpha.gif", fps=10)
gif(Hx_animation, "Hx_3D_pml_nx_$(nx)_ny_$(ny)_nz_$(nz)_alpha_$alpha.gif", fps=10)
gif(Hy_animation, "Hy_3D_pml_nx_$(nx)_ny_$(ny)_nz_$(nz)_alpha_$alpha.gif", fps=10)
gif(Hz_animation, "Hz_3D_pml_nx_$(nx)_ny_$(ny)_nz_$(nz)_alpha_$alpha.gif", fps=10)
