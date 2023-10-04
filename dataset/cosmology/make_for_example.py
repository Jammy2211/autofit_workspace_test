import numpy as np

import autolens as al
import autolens.plot as aplt

data = al.Array2D.from_fits(
    file_path="dataset/cosmology/data_in.fits", pixel_scales=0.05
)

mask = al.Mask2D.circular(
    shape_native=data.shape_native, pixel_scales=data.pixel_scales, radius=3.0
)

zoom_shape = mask.zoom_shape_native

data = data.apply_mask(mask=mask)
data = data.resized_from(new_shape=zoom_shape)

np.save(file="dataset/cosmology/data.npy", arr=data.native)

output = aplt.Output(path="dataset/cosmology", filename="data", format="png")

array_plotter = aplt.Array2DPlotter(
    array=data, mat_plot_2d=aplt.MatPlot2D(output=output)
)
array_plotter.figure_2d()


noise_map = al.Array2D.from_fits(
    file_path="dataset/cosmology/noise_map_in.fits", pixel_scales=0.05
)

noise_map = noise_map.apply_mask(mask=mask)
noise_map = noise_map.resized_from(new_shape=zoom_shape)

np.save(file="dataset/cosmology/noise_map.npy", arr=noise_map.native)

output = aplt.Output(path="dataset/cosmology", filename="noise_map", format="png")

array_plotter = aplt.Array2DPlotter(
    array=noise_map, mat_plot_2d=aplt.MatPlot2D(output=output)
)
array_plotter.figure_2d()


psf = al.Kernel2D.from_fits(
    file_path="dataset/cosmology/psf_in.fits", hdu=0, pixel_scales=0.05
)

psf = psf.resized_from(new_shape=(7, 7))

np.save(file="dataset/cosmology/psf.npy", arr=psf.native)

output = aplt.Output(path="dataset/cosmology", filename="psf", format="png")

array_plotter = aplt.Array2DPlotter(
    array=psf, mat_plot_2d=aplt.MatPlot2D(output=output)
)
array_plotter.figure_2d()


mask = al.Mask2D.circular(
    shape_native=(121, 121), pixel_scales=data.pixel_scales, radius=3.0
)


grid = al.Grid2D.from_mask(mask=mask)

np.save(file="dataset/cosmology/grid.npy", arr=grid.native)
