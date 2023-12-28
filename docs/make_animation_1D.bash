figs_path="."
imgs_path="./viz_out_1D"
convert -delay 20 -loop 0 $imgs_path/*.png "$figs_path/Maxwell_1D_xpu.gif"