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

gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

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
ue3_instances = [samp.error_instance_at_upper_sigma(sigma=3.0) for samp in samples_list]
le3_instances = [samp.error_instance_at_lower_sigma(sigma=3.0) for samp in samples_list]

mp_centres = [instance.gaussian.centre for instance in mp_instances]
ue3_centres = [instance.gaussian.centre for instance in ue3_instances]
le3_centres = [instance.gaussian.centre for instance in le3_instances]

print(f"Median PDF inferred centre values")
print(mp_centres)
print()


"""
__Likelihood Check__
"""
log_likelihood = sum([max(samples.log_likelihood_list) for samples in samples_list])

print(f"Overall Likelihood = {log_likelihood}")
print()


"""
__Overall Gaussian__
"""
ue1_instances = [samp.instance_at_upper_sigma(sigma=1.0) for samp in samples_list]
le1_instances = [samp.instance_at_lower_sigma(sigma=1.0) for samp in samples_list]

ue1_centres = [instance.gaussian.centre for instance in ue1_instances]
le1_centres = [instance.gaussian.centre for instance in le1_instances]

error_list = [ue1 - le1 for ue1, le1 in zip(ue1_centres, le1_centres)]


class Analysis(af.Analysis):
    def __init__(self, data, errors):

        super().__init__()

        self.data = np.array(data)
        self.errors = np.array(errors)

    def log_likelihood_function(self, instance):

        summand1 = np.sum(
            -np.divide(
                (self.data - instance.median) ** 2,
                2 * (instance.scatter ** 2 + self.errors ** 2),
            )
        )
        summand2 = -np.sum(0.5 * np.log(instance.scatter ** 2 + self.errors ** 2))

        return summand1 + summand2


class SampleDist:
    def __init__(
        self,
        median=0.0,  # <- **PyAutoFit** recognises these constructor arguments are the Gaussian`s model parameters.
        scatter=0.01,
    ):

        self.median = median
        self.scatter = scatter

    def probability_from_values(self, values):

        values = np.sort(np.array(values))
        transformed_values = np.subtract(values, self.median)

        return np.multiply(
            np.divide(1, self.scatter * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_values, self.scatter))),
        )


model = af.Model(SampleDist)

model.median = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.scatter = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

analysis = Analysis(data=mp_centres, errors=error_list)
search = af.DynestyStatic(nlive=100)

result = search.fit(model=model, analysis=analysis)


samples = result.samples


median = samples.median_pdf_instance.median

u1_error = samples.instance_at_upper_sigma(sigma=1.0).median
l1_error = samples.instance_at_lower_sigma(sigma=1.0).median

u3_error = samples.instance_at_upper_sigma(sigma=3.0).median
l3_error = samples.instance_at_lower_sigma(sigma=3.0).median

print(
    f"Inferred value of the hierarchical median via simple fit to {total_datasets} datasets: \n "
)
print(f"{median} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{median} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")
print()


scatter = samples.median_pdf_instance.scatter

u1_error = samples.instance_at_upper_sigma(sigma=1.0).scatter
l1_error = samples.instance_at_lower_sigma(sigma=1.0).scatter

u3_error = samples.instance_at_upper_sigma(sigma=3.0).scatter
l3_error = samples.instance_at_lower_sigma(sigma=3.0).scatter

print(
    f"Inferred value of the hierarchical scatter via simple fit to {total_datasets} datasets: \n "
)
print(f"{scatter} ({l1_error} {u1_error}) [1.0 sigma confidence intervals]")
print(f"{scatter} ({l3_error} {u3_error}) [3.0 sigma confidence intervals]")
print()
