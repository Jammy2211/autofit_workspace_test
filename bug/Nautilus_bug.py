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

gaussian_7 = af.Model(af.ex.Gaussian)

gaussian_7.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_7.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_7.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_8 = af.Model(af.ex.Gaussian)

gaussian_8.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_8.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_8.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_9 = af.Model(af.ex.Gaussian)

gaussian_9.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_9.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_9.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

gaussian_10 = af.Model(af.ex.Gaussian)

gaussian_10.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian_10.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
gaussian_10.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1, gaussian_2=gaussian_2,
                      gaussian_3=gaussian_3, gaussian_4=gaussian_4, gaussian_5=gaussian_5,
                      gaussian_6=gaussian_6, gaussian_7=gaussian_7, gaussian_8=gaussian_8,
                      gaussian_9=gaussian_9, gaussian_10=gaussian_10)

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
    n_live=100,  # Number of so-called live points. New bounds are constructed so that they encompass the live points.
    n_update=None,  # The maximum number of additions to the live set before a new bound is created
    enlarge_per_dim=1.1,  # Along each dimension, outer ellipsoidal bounds are enlarged by this factor.
    n_points_min=None,  # The minimum number of points each ellipsoid should have. Effectively, ellipsoids with less than twice that number will not be split further.
    split_threshold=100,  # Threshold used for splitting the multi-ellipsoidal bound used for sampling.
    n_networks=4,  # Number of networks used in the estimator.
    n_batch=100,  # Number of likelihood evaluations that are performed at each step. If likelihood evaluations are parallelized, should be multiple of the number of parallel processes.
    n_like_new_bound=None,  # The maximum number of likelihood calls before a new bounds is created. If None, use 10 times n_live.
    vectorized=False,  # If True, the likelihood function can receive multiple input sets at once.
    seed=None,  # Seed for random number generation used for reproducible results accross different runs.
    f_live=1e-15,  # Maximum fraction of the evidence contained in the live set before building the initial shells terminates.
    n_shell=1,  # Minimum number of points in each shell. The algorithm will sample from the shells until this is reached. Default is 1.
    n_eff=500,  # Minimum effective sample size. The algorithm will sample from the shells until this is reached. Default is 10000.
    discard_exploration=False,  # Whether to discard points drawn in the exploration phase. This is required for a fully unbiased posterior and evidence estimate.
    verbose=True,  # Whether to print information about the run.
    n_like_max=np.inf,  # Maximum number of likelihood evaluations. Regardless of progress, the sampler will stop if this value is reached. Default is infinity.
    iterations_per_update=10,
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. Lets use it to
compare the maximum log likelihood `Gaussian` to the data.
"""
model_data = result.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
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
plt.plot(range(data.shape[0]), model_data, color="r")
plt.title("Nautilus model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
__Search Internal__

The result also contains the internal representation of the non-linear search.

The internal representation of the non-linear search ensures that all sampling info is available in its native form.
This can be passed to functions which take it as input, for example if the sampling package has bespoke visualization 
functions.

For `DynestyStatic`, this is an instance of the `Sampler` object (`from nautilus import Sampler`).
"""
search_internal = result.search_internal

print(search_internal)

"""
The internal search is by default not saved to hard-disk, because it can often take up quite a lot of hard-disk space
(significantly more than standard output files).

This means that the search internal will only be available the first time you run the search. If you rerun the code 
and the search is bypassed because the results already exist on hard-disk, the search internal will not be available.

If you are frequently using the search internal you can have it saved to hard-disk by changing the `search_internal`
setting in `output.yaml` to `True`. The result will then have the search internal available as an attribute, 
irrespective of whether the search is re-run or not.
"""