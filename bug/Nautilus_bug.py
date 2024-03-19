"""
Searches=Nautilus
=======================

This example illustrates how to use the nested sampling algorithm Nautilus.

Information about Nautilus can be found at the following links:

 - https://nautilus-sampler.readthedocs.io/en/stable/index.html
 - https://github.com/johannesulf/nautilus
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path
import os

cwd = os.getcwd()
from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "bug"))


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

plt.errorbar(
    x=range(data.shape[0]),
    y=data,
    yerr=noise_map,
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.show()
plt.close()

"""
__Model + Analysis__

We create the model and analysis, which in this example is a single `Gaussian` and therefore has dimensionality N=3.
"""
gaussian_0 = af.Model(af.ex.Gaussian)

gaussian_0.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_0.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_0.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_1 = af.Model(af.ex.Gaussian)

gaussian_1.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_1.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_1.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_2 = af.Model(af.ex.Gaussian)

gaussian_2.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_2.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_2.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_3 = af.Model(af.ex.Gaussian)

gaussian_3.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_3.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_3.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_4 = af.Model(af.ex.Gaussian)

gaussian_4.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_4.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_4.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_5 = af.Model(af.ex.Gaussian)

gaussian_5.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_5.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_5.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_6 = af.Model(af.ex.Gaussian)

gaussian_6.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_6.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_6.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1, gaussian_2=gaussian_2,
                      gaussian_3=gaussian_3, gaussian_4=gaussian_4, gaussian_5=gaussian_5,
                      gaussian_6=gaussian_6)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Search__

We now create and run the `Nautilus` object which acts as our non-linear search. 

We manually specify all of the Nautilus settings, descriptions of which are provided at the following webpage:

https://github.com/johannesulf/nautilus
"""
search = af.Nautilus(
    path_prefix=path.join("searches"),
    name="Nautilus",
    number_of_cores=4,
    n_live=500,  # Number of so-called live points. New bounds are constructed so that they encompass the live points.
    f_live=1e-15,  # Maximum fraction of the evidence contained in the live set before building the initial shells terminates.
    iterations_per_update=10000,
)

result = search.fit(model=model, analysis=analysis)

