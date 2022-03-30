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

import autofit as af

total_gaussians = 10

"""
__Gaussian x1 low snr (centre fixed to 50.0)__

This is used for demonstrating expectation propagation, whereby a shared `centre` parameter is inferred from a sample 
of `total_gaussians` 1D Gaussian datasets.
"""
for i in range(total_gaussians):

    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1__low_snr", f"dataset_{i}"
    )
    gaussian = af.ex.Gaussian(centre=50.0, normalization=0.5, sigma=5.0)
    util.simulate_dataset_1d_via_gaussian_from(
        gaussian=gaussian, dataset_path=dataset_path
    )


total_gaussians = 10

"""
__Gaussian x1 low snr (centre drawn from parent Gaussian distribution to 50.0)__

This is used for demonstrating expectation propagation and hierachical modeling, whereby a the `centre` parameters 
of a sample of `total_gaussians` 1D Gaussian datasets are drawn from a Gaussian distribution.
"""
gaussian_parent_model = af.Model(
    af.ex.Gaussian,
    centre=af.GaussianPrior(mean=50.0, sigma=10.0, lower_limit=0.0, upper_limit=100.0),
    normalization=0.5,
    sigma=5.0,
)

for i in range(total_gaussians):

    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1__hierarchical", f"dataset_{i}"
    )

    gaussian = gaussian_parent_model.random_instance()

    util.simulate_dataset_1d_via_gaussian_from(
        gaussian=gaussian, dataset_path=dataset_path
    )
