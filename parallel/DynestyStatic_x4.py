"""
Modeling: Mass Total + Source Parametric
========================================

This script gives a profile of a `DynestyStatic` model-fit to an `Imaging` dataset where the lens model is initialized,
where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear`.
 - The source galaxy's light is a parametric `EllSersic`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "searches"))

import autofit as af

"""
__Paths__
"""
dataset_name = "gaussian_x1"
path_prefix = path.join("parallel")

"""
__Search__
"""
search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="DynestyStatic_x4",
    unique_tag=dataset_name,
    nlive=250,
    walks=10,
    iterations_per_update=100000,
    number_of_cores=4,
)

"""
__Data__

This example fits a single 1D Gaussian, we therefore load and plot data containing one Gaussian.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
__Model + Analysis__

We create the model and analysis, which in this example is a single `Gaussian` and therefore has dimensionality N=3.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)


result = search.fit(model=model, analysis=analysis)

"""
Finished.
"""
