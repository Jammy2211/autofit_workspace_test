"""
Searches: DynestyStatic
=======================

This example illustrates how to use the nested sampling algorithm DynestyStatic.

Information about Dynesty can be found at the following links:

 - https://github.com/joshspeagle/dynesty
 - https://dynesty.readthedocs.io/en/latest/
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autofit as af

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
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search__

We now create and run the `DynestyStatic` object which acts as our non-linear search. 

We manually specify all of the Dynesty settings, descriptions of which are provided at the following webpage:

 https://dynesty.readthedocs.io/en/latest/api.html
 https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.nestedsamplers
"""
search = af.DynestyStatic(
    path_prefix="chaining_limits",
    name="search[1]",
)

result = search.fit(model=model, analysis=analysis)


model = result.model

search = af.DynestyStatic(
    path_prefix="chaining_limits",
    name="search[2]",
)

result = search.fit(model=model, analysis=analysis)
