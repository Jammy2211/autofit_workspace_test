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
"""
We use a `Model` to create and customize the `Gaussian` and `Exponential` models.
"""
gaussian_0 = af.Model(af.ex.Gaussian)
gaussian_1 = af.Model(af.ex.Gaussian)

# gaussian_1.add_assertion(
#     serial_trap_1.release_timescale < serial_trap_2.release_timescale
# )

"""
Checkout `autofit_workspace/config/priors/model.json`, this config file defines the default priors of the `Gaussian` 
and `Exponential` model components. 

We can manually customize the priors of the model used by the non-linear search.
"""
gaussian_0.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_0.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)
gaussian_1.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_1.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=1e2)

gaussian_0.sigma = af.UniformPrior(lower_limit=0.5, upper_limit=1.0)
gaussian_1.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_0.add_assertion(gaussian_0.sigma < 0.5)

"""
We can now compose the overall model using a `Collection`, which takes the model components we defined above.
"""
model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search__

We now create and run the `DynestyStatic` object which acts as our non-linear search. 

We manually specify all of the Dynesty settings, descriptions of which are provided at the following webpage:

 https://dynesty.readthedocs.io/en/latest/api.html
 https://dynesty.readthedocs.io/en/latest/api.html#module-dynesty.nestedsamplers
"""
search = af.DynestyStatic(
    name="initializer_assertion",
    nlive=50,
)

result = search.fit(model=model, analysis=analysis)
