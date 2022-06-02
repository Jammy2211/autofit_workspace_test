"""
Fit every 1D Gaussian with a shared centre individually and estimate the centre at the end of the analysis.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
from os import path
import sys

workspace_path = os.getcwd()
plot_path = path.join(workspace_path, "paper", "images", "individual_pdf")

import autofit as af
import autofit.plot as aplt


"""
__Data__

We quickly set up the name of each dataset, which is used below for loading the datasets.
"""
total_datasets = int(sys.argv[1])

dataset_name_list = []

for dataset_index in range(total_datasets):

    dataset_name_list.append(f"dataset_{dataset_index}")


signal_to_noise_ratio_list = [5.0, 25.0, 100.0]
signal_to_noise_ratio = signal_to_noise_ratio_list[int(sys.argv[2])]

"""
__Model__
"""
gaussian = af.Model(af.ex.Gaussian)

gaussian.centre = af.GaussianPrior(mean=50.0, sigma=30.0, lower_limit=0.0, upper_limit=100.0)
gaussian.normalization = af.GaussianPrior(mean=10.0, sigma=10.0, lower_limit=0.0)
gaussian.sigma = af.GaussianPrior(
            lower_limit=0.0, upper_limit=20.0, mean=10.0, sigma=10.0
        )

model = af.Collection(gaussian=gaussian)

"""
__Model Fits (one-by-one)__

For every dataset we now create an `Analysis` class using it and use `Dynesty` to fit it with a `Gaussian`.

The `Result` is stored in the list `results`.
"""
result_list = []

for dataset_name in dataset_name_list:

    """
    Load the dataset from the `autofit_workspace/dataset` folder.
    """
    dataset_path = path.join(
        "dataset", f"gaussian_x1__snr_{signal_to_noise_ratio}", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    """
    Create the `DynestyStatic` non-linear search and use it to fit the data.
    """
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    dynesty = af.DynestyStatic(
        name="",
        path_prefix=path.join(f"individual__{signal_to_noise_ratio}"),
        unique_tag=dataset_name,
        nlive=200,
        dlogz=1e-4,
        sample="rwalk",
        walks=10,
    )

    result_list.append(dynesty.fit(model=model, analysis=analysis))

"""
__Results__

Checkout the output folder, you should see three new sets of results corresponding to our 3 `Gaussian` datasets.

In the `model.results` file of each fit, it will be clear that the `centre` value of every fit (and the other 
parameters) have much larger errors than other **PyAutoFit** examples due to the low signal to noise of the data.

The `result_list` allows us to plot the median PDF value and 3.0 confidence intervals of the `centre` estimate from
the model-fit to each dataset.
"""
samples_list = [result.samples for result in result_list]

mp_instances = [samps.median_pdf_instance for samps in samples_list]
mp_centres = [instance.gaussian.centre for instance in mp_instances]

print(f"Median PDF inferred centre values")
print(mp_centres)
print()

"""
__Estimating the Centre (Weighted Average)__
"""
samples_list = [result.samples for result in result_list]

ue1_instances = [samp.instance_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.instance_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_centres = [instance.gaussian.centre for instance in ue1_instances]
le1_centres = [instance.gaussian.centre for instance in le1_instances]

error_list = [ue1 - le1 for ue1, le1 in zip(ue1_centres, le1_centres)]

values = np.asarray(mp_centres)
sigmas = np.asarray(error_list)

weights = 1 / sigmas**2.0
weight_averaged = np.sum(1.0 / sigmas**2)

weighted_centre = np.sum(values * weights) / np.sum(weights, axis=0)
weighted_error = 1.0 / np.sqrt(weight_averaged)

print(f"{weighted_centre} ({weighted_error}) [1.0 sigma confidence intervals]")

"""
__Likelihood Check__
"""
log_likelihood = sum([max(samples.log_likelihood_list) for samples in samples_list])

print(f"Overall Likelihood = {log_likelihood}")
print()

# """
# __PDFs__
# """
# from getdist import MCSamples
# from getdist import plots
# import matplotlib.pyplot as plt
#
# print(samples_list[0].model.parameter_labels_with_superscripts_latex)
#
# gd_samples_list = [MCSamples(
#     samples=np.asarray(samples.parameter_lists),
#     loglikes=np.asarray(samples.log_likelihood_list),
#     weights=np.asarray(samples.weight_list),
#     names=samples.model.parameter_names,
#     labels=samples.model.parameter_labels_with_superscripts_latex,
#     sampler="nested",
# ) for samples in samples_list]
#
# gd_plotter = plots.get_subplot_plotter(width_inch=12)
#
# gd_plotter.triangle_plot(roots=gd_samples_list, filled=False)
#
# plt.savefig(path.join(plot_path, f"pdf_{signal_to_noise_ratio}_{total_datasets}.png"))
# plt.close()