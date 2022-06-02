"""
Fit every 1D Gaussian with a shared centre simultaneously, with the centre parameter identical across all
datasets.
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
plot_path = path.join(workspace_path, "paper", "images", "simultaneous_pdf")

import autofit as af
import autofit.plot as aplt

"""
__Dataset__

For each dataset we now set up the correct path and load it. 

Whereas in the previous tutorial we fitted each dataset one-by-one, in this tutorial we instead store each dataset 
in a list so that we can set up a single model-fit that fits the 5 datasets simultaneously.
"""
total_datasets = int(sys.argv[1])

dataset_name_list = []
data_list = []
noise_map_list = []

signal_to_noise_ratio_list = [5.0, 25.0, 100.0]
signal_to_noise_ratio = signal_to_noise_ratio_list[int(sys.argv[2])]

for dataset_index in range(total_datasets):

    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", f"gaussian_x1__snr_{signal_to_noise_ratio}", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    dataset_name_list.append(dataset_name)
    data_list.append(data)
    noise_map_list.append(noise_map)

"""
__Analysis__

For each dataset we now create a corresponding `Analysis` class. 
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)

"""
__Model__

We now compose the graphical model that we fit, using the `Model` object you are now familiar with.

We begin by setting up a shared prior for `centre`. 

We set up this up as a single `GaussianPrior` which is passed to separate `Model`'s for each `Gaussian` below.
"""
centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0, lower_limit=0.0, upper_limit=100.0)

"""
We now set up three `Model`'s, each of which contain a `Gaussian` that is used to fit each of the 
datasets we loaded above.

All three of these `Model`'s use the `centre_shared_prior`. This means all three model-components use 
the same value of `centre` for every model composed and fitted by the `NonLinearSearch`, reducing the dimensionality 
of parameter space from N=9 (e.g. 3 parameters per Gaussian) to N=7.
"""
model_list = []

for model_index in range(len(data_list)):

    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!
    gaussian.normalization = af.GaussianPrior(mean=10.0, sigma=10.0, lower_limit=0.0)
    gaussian.sigma = af.GaussianPrior(
            lower_limit=0.0, upper_limit=20.0, mean=10.0, sigma=10.0
        )

    model = af.Collection(gaussian=gaussian)

    model_list.append(model)

"""
__Analysis Factors__

Above, we composed a model consisting of three `Gaussian`'s with a shared `centre` prior. We also loaded three datasets
which we intend to fit with each of these `Gaussians`, setting up each in an `Analysis` class that defines how the 
model is used to fit the data.

We now simply pair each model-component to each `Analysis` class, so that **PyAutoFit** knows that: 

- `gaussian_0` fits `data_0` via `analysis_0`.
- `gaussian_1` fits `data_1` via `analysis_1`.
- `gaussian_2` fits `data_2` via `analysis_2`.

The point where a `Model` and `Analysis` class meet is called an `AnalysisFactor`. 

This term is used to denote that we are composing a graphical model, which is commonly termed a 'factor graph'. A 
factor defines a node on this graph where we have some data, a model, and we fit the two together. The 'links' between 
these different nodes then define the global model we are fitting.
"""
search = af.DynestyStatic(
    nlive=300,
    dlogz=1e-4,
    sample="rwalk",
    walks=10,
)

analysis_factor_list = []

dataset_index = 0

for model, analysis in zip(model_list, analysis_list):

    dataset_name = f"dataset_{dataset_index}"
    dataset_index += 1

    analysis_factor = af.AnalysisFactor(
        prior_model=model, analysis=analysis, optimiser=search, name=dataset_name
    )

    analysis_factor_list.append(analysis_factor)

"""
__Factor Graph__

We combine our `AnalysisFactor`'s into one, to compose a factor graph.

So, what is a factor graph?

A factor graph defines the graphical model we have composed. For example, it defines the different model components 
that make up our model (e.g. the three `Gaussian` classes) and how their parameters are linked or shared (e.g. that
each `Gaussian` has its own unique `normalization` and `centre`, but a shared `sigma` parameter.

This is what our factor graph looks like: 

The factor graph above is made up of two components:

- Nodes: these are points on the graph where we have a unique set of data and a model that is made up of a subset of 
our overall graphical model. This is effectively the `AnalysisFactor` objects we created above. 

- Links: these define the model components and parameters that are shared across different nodes and thus retain the 
same values when fitting different datasets.
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
__Search__

We can now create a non-linear search and used it to the fit the factor graph, using its `global_prior_model` property.
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(
    name=f"ep_{signal_to_noise_ratio}",
)

factor_graph_result = factor_graph.optimise(
    optimiser=laplace, paths=paths, ep_history=af.EPHistory(kl_tol=0.05), max_steps=5
)

"""
__Output__

The MeanField object representing the posterior.
"""
mean_field = factor_graph_result.updated_ep_mean_field.mean_field
print(mean_field)
print()

print(mean_field.variables)
print()

"""
The logpdf of the posterior at the point specified by the dictionary values
"""
# factor_graph_result.updated_ep_mean_field.mean_field(values=None)
print()

"""
A dictionary of the mean with variables as keys.
"""
print(f"Centre Mean = {mean_field.mean[centre_shared_prior]}")
print()

"""
A dictionary of the variance with variables as keys.
"""
print(f"Centre Variance = {mean_field.variance[centre_shared_prior]}")
print()

"""
A dictionary of the variance with variables as keys.
"""
print(f"Centre 1 Sigma = {np.sqrt(mean_field.variance[centre_shared_prior])}")
print()

"""
A dictionary of the s.d./variance**0.5 with variables as keys.
"""
print(f"Centre SD/sqrt(variance) = {mean_field.scale[centre_shared_prior]}")
print()

"""
Exmaple using original model.
"""
print(f"Normalization Gaussian Dataset 0 Mean = {mean_field.mean[model_list[0].gaussian.normalization]}")

"""
self.updated_ep_mean_field.mean_field[v: Variable] gives the Message/approximation of the posterior for an individual variable of the model
"""
# factor_graph_result.updated_ep_mean_field.mean_field["help"]

"""
Finish.
"""