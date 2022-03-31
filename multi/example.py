"""
Modeling: Mass Total + Source Parametric
========================================

This script fits a multi-wavelength `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `EllSersic`.

Two images are fitted, corresponds to a greener ('g' band) redder image (`r` band).

This is an advanced script and assumes previous knowledge of the core **PyAutoLens** API for simulating images. Thus,
certain parts of code are not documented to ensure the script is concise.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).

The strings are used for load each dataset.
"""
color_list = ["g", "r"]

"""
__Pixel Scales__

Every multi-wavelength dataset can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.012]

"""
__Dataset__

Load and plot each multi-wavelength strong lens dataset, using a list of their waveband colors.
"""
dataset_type = "imaging"
dataset_label = "multi"
dataset_name = "mass_sie__source_sersic"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

imaging_list = [
    al.Imaging.from_fits(
        image_path=path.join(dataset_path, f"{color}_image.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

for imaging in imaging_list:
    imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
    imaging_plotter.subplot_imaging()

"""
__Masking__

The model-fit requires a `Mask2D` defining the regions of the image we fit the lens model to the data, which we define
and use to set up the `Imaging` object that the lens model fits.

For multi-wavelength lens modeling, we should use the same mask for every dataset whenever possible. This is not
absolutely necessary, but provides a more reliable analysis.
"""
mask_list = [
    al.Mask2D.circular(
        shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
    )
    for imaging in imaging_list
]

imaging_list = [
    imaging.apply_mask(mask=mask) for imaging, mask in zip(imaging_list, mask_list)
]

for imaging in imaging_list:
    imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
    imaging_plotter.subplot_imaging()

"""
__Analysis__

We create an `Analysis` object for every dataset.
"""
analysis_list = [al.AnalysisImaging(dataset=imaging) for imaging in imaging_list]

"""
By summing this list of analysis objects, we create an overall `CombinedAnalysis` which we can use to fit the 
multi-wavelength imaging data, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis objects (e.g. the fit to each separate waveband).

 - The summing process ensures that tasks such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis and therefore each dataset.

 - Next, we will use this combined analysis to parameterize a model where certain lens parameters vary across
 the dataset.
"""
analysis = sum(analysis_list)

"""
We can parallelize the likelihood function of these analysis classes, whereby each evaluation is performed on a 
different CPU.
"""
analysis.n_cores = 2

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear` [7 parameters].

 - The source galaxy's light is a parametric `EllSersic`, where the `intensity` parameter of the source galaxy
 for each individual waveband of imaging is a different free parameter [8 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=15.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal, shear=al.mp.ExternalShear
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
We now make the intensity a free parameter across every analysis object.
"""
analysis = analysis.with_free_parameters(model.galaxies.source.bulge.intensity)

"""
__Search__
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "multi"),
    name="mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    nlive=50,
    number_of_cores=1,
)

"""
__Model-Fit__
"""
result_list = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by this model-fit is a list of `Result` objects, because we used a combined analysis.
Each result corresponds to each analysis, and therefore corresponds to the model-fit at that wavelength.

For example, close inspection of the `max_log_likelihood_instance` of the two results showns that all parameters,
except the `intensity` of the source galaxy's `bulge`, are identical.
"""
print(result_list[0].max_log_likelihood_instance)
print(result_list[1].max_log_likelihood_instance)

"""
Plotting each result's tracer shows that the source appears different, owning to its different intensities.
"""
for result in result_list:
    tracer_plotter = aplt.TracerPlotter(
        tracer=result.max_log_likelihood_tracer, grid=result.grid
    )
    tracer_plotter.subplot_tracer()

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
    fit_imaging_plotter.subplot_fit_imaging()

"""
The `Samples` object still has the dimensions of the overall non-linear search (in this case N=15). 

Therefore, the samples is identical in every result object.
 """
for result in result_list:
    dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
    dynesty_plotter.cornerplot()

"""
Checkout `autolens_workspace/notebooks/results` for a full description of analysing results in **PyAutoLens**.
"""
