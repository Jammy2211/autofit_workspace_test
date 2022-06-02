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

import util
from os import path
import numpy as np
import sys

import autofit as af

total_datasets = int(sys.argv[1])

"""
__Gaussian x1 low snr (centre drawn from parent Gaussian distribution to 50.0)__

This is used for demonstrating expectation propagation and hierachical modeling, whereby a the `centre` parameters
of a sample of `total_datasets` 1D Gaussian datasets are drawn from a Gaussian distribution.
"""

for signal_to_noise_ratio in [5.0, 25.0, 100.0]:

    for i in range(total_datasets):

        centre_prior = af.GaussianPrior(
            mean=50.0, sigma=10.0, lower_limit=0.0, upper_limit=100.0
        )
        while True:
            try:
                centre_value = centre_prior.value_for(unit=np.random.random(1))
                break
            except af.exc.PriorLimitException:
                continue

        sigma_prior = af.GaussianPrior(
            lower_limit=0.0, upper_limit=20.0, mean=10.0, sigma=10.0
        )
        while True:
            try:
                sigma_value = sigma_prior.value_for(unit=np.random.random(1))
                break
            except af.exc.PriorLimitException:
                continue

        gaussian_parent_model = af.Model(
            af.ex.Gaussian, centre=centre_value, normalization=1.0, sigma=sigma_value
        )

        dataset_path = path.join(
            "dataset", f"gaussian_x1__snr_{signal_to_noise_ratio}", f"dataset_{i}"
        )

        gaussian = gaussian_parent_model.random_instance()

        util.simulate_dataset_1d_via_gaussian_from(
            gaussian=gaussian, dataset_path=dataset_path
        )
