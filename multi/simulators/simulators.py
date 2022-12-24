"""
__Simulators__

These scripts simulate the 1D Gaussian datasets used to demonstrate model-fitting.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import util
from os import path

import autofit as af

"""
__Gaussian x1 (Identical 0)__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_identical_0")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (Identical 1)__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_identical_1")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
__Gaussian x1 (Identical 2)__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1_identical_2")
gaussian = af.ex.Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
util.simulate_dataset_1d_via_gaussian_from(gaussian=gaussian, dataset_path=dataset_path)

"""
Finish.
"""
