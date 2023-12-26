figs_path="."
imgs_path="./viz_out"
convert -delay 5 -loop 0 $imgs_path/*.png "$figs_path/Maxwell_2D_xpu.gif"