"""
Feature: Interpolator
=====================
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import numpy as np

import autofit as af
import autofit.plot as aplt

"""
__Dataset Names__

"""
dataset_name_list = ["gaussian_x1_t0", "gaussian_x1_t1", "gaussian_x1_t2"]

time_list = [0, 1, 2]

"""
__Model__

Next, we create our model, which again corresponds to a single `Gaussian` with manual priors.
"""

instance_list = []

for dataset_name, time in zip(dataset_name_list, time_list):
    """
    The code below loads the dataset and sets up the Analysis class.
    """
    dataset_path = path.join("interpolator", "dataset", dataset_name)

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    model = af.Collection(gaussian=af.ex.Gaussian, time=time)

    model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.gaussian.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.gaussian.sigma = af.GaussianPrior(
        mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
    )

    search = af.DynestyStatic(
        name=f"model_t{time}",
        path_prefix=path.join("interpolator"),
        unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
        nlive=50,
    )

    print(
        f"The non-linear search has begun running. This Jupyter notebook cell with progress once search has completed, "
        f"this could take a "
        f"few minutes!"
    )

    result = search.fit(model=model, analysis=analysis)

    instance_list.append(result.instance)

interpolator = af.LinearInterpolator(instances=instance_list)

instance = interpolator[interpolator.time == 1.5]

print(instance.gaussian.centre)
