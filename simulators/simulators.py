"""
__Simulators__

These scripts simulate the 1D Gaussian datasets used to demonstrate model-fitting.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import profiles
import util
from os import path

"""
__Gaussian x1__
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
util.simulate_line_from_gaussian(gaussian=gaussian, dataset_path=dataset_path)
