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
__Gaussian x1 low snr (centre fixed to 50.0)__

This is used for demonstrating expectation propagation, whereby a shared `centre` parameter is inferred from a sample 
of `total_datasets` 1D Gaussian datasets.
"""

for signal_to_noise_ratio in [5.0, 25.0, 100.0]:

    for i in range(total_datasets):

        dataset_path = path.join(
            "dataset",
            f"gaussian_x1__snr_{signal_to_noise_ratio}",
            f"dataset_{i}",
        )

        sigma_prior = af.GaussianPrior(
            lower_limit=0.0, upper_limit=20.0, mean=10.0, sigma=10.0
        )
        while True:
            try:
                sigma_value = sigma_prior.value_for(unit=np.random.random(1))
                break
            except af.exc.PriorLimitException:
                continue

        gaussian = af.ex.Gaussian(centre=50.0, normalization=1.0, sigma=sigma_value)

        util.simulate_dataset_1d_via_gaussian_from(
            gaussian=gaussian,
            dataset_path=dataset_path,
            signal_to_noise_ratio=signal_to_noise_ratio,
        )
