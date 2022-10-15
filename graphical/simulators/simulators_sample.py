"""
__Simulators__

These scripts simulates many 1D Gaussian datasets with a low signal to noise ratio, which are used to demonstrate
model-fitting.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af
import util

"""
__Gaussian x1 low snr (centre fixed to 50.0)__

This is used for demonstrating expectation propagation, whereby a shared `centre` parameter is inferred from a sample 
of `total_datasets` 1D Gaussian datasets.
"""
total_datasets = 2

for i in range(total_datasets):

    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1__sample", f"dataset_{i}"
    )
    gaussian = af.ex.Gaussian(centre=50.0, normalization=0.5, sigma=5.0)
    util.simulate_dataset_1d_via_gaussian_from(
        gaussian=gaussian, dataset_path=dataset_path
    )
