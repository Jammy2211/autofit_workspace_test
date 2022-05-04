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
import util
from os import path

import autofit as af

"""
__Gaussian x1 low snr (centre fixed to 50.0)__

This is used for demonstrating expectation propagation, whereby a shared `centre` parameter is inferred from a sample 
of `total_datasets` 1D Gaussian datasets.
"""
total_datasets = 10
signal_to_noise_ratio = 5.0

for i in range(total_datasets):

    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1__sample", f"dataset_{i}__snr_{signal_to_noise_ratio}"
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

"""
__Gaussian x1 low snr (centre drawn from parent Gaussian distribution to 50.0)__

This is used for demonstrating expectation propagation and hierachical modeling, whereby a the `centre` parameters 
of a sample of `total_datasets` 1D Gaussian datasets are drawn from a Gaussian distribution.
"""
total_datasets = 10
signal_to_noise_ratio = 5.0

gaussian_parent_model = af.Model(
    af.ex.Gaussian,
    centre=af.GaussianPrior(mean=50.0, sigma=10.0, lower_limit=0.0, upper_limit=100.0),
    normalization=0.5,
    sigma=5.0,
)

for i in range(total_datasets):

    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1__hierarchical", f"dataset_{i}__snr_{signal_to_noise_ratio}"
    )

    gaussian = gaussian_parent_model.random_instance()

    util.simulate_dataset_1d_via_gaussian_from(
        gaussian=gaussian, dataset_path=dataset_path, signal_to_noise_ratio=signal_to_noise_ratio
    )
