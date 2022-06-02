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
__Model Individual Factors__

We first set up a model for each `Gaussian` which is individually fitted to each 1D dataset, which forms the
factors on the factor graph we compose. 

This uses a nearly identical for loop to the previous tutorial, however a shared `centre` is no longer used and each 
`Gaussian` is given its own prior for the `centre`. 

We will see next how this `centre` is used to construct the hierachical model.
"""
model_list = []

for model_index in range(len(data_list)):

    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

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
search = af.DynestyStatic(nlive=300, dlogz=1e-4, sample="rwalk", walks=10)

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
__Model__

We now compose the hierarchical model that we fit, using the individual Gaussian model components created above.

We first create a `HierarchicalFactor`, which represents the parent Gaussian distribution from which we will assume 
that the `centre` of each individual `Gaussian` dataset is drawn. 

For this parent `Gaussian`, we have to place priors on its `mean` and `sigma`, given that they are parameters in our
model we are ultimately fitting for.
"""

# hierarchical_factor = af.HierarchicalFactor(
#     af.GaussianPrior,
#     mean=af.GaussianPrior(mean=50.0, sigma=10, lower_limit=0.0, upper_limit=100.0),
#     sigma=af.GaussianPrior(mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0),
# )

hierarchical_factor = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.UniformPrior(lower_limit=0.0, upper_limit=100.0),
    sigma=af.UniformPrior(lower_limit=0.0, upper_limit=100.0),
)

"""
We now add each of the individual model `Gaussian`'s `centre` parameters to the `hierarchical_factor`.

This composes the hierarchical model whereby the individual `centre` of every `Gaussian` in our dataset is now assumed 
to be drawn from a shared parent distribution. It is the `mean` and `sigma` of this distribution we are hoping to 
estimate.
"""

for model in model_list:

    hierarchical_factor.add_drawn_variable(model.gaussian.centre)

"""
__Factor Graph__

We now create the factor graph for this model, using the list of `AnalysisFactor`'s and the hierarchical factor.

Note that in previous tutorials, when we created the `FactorGraphModel` we only passed the list of `AnalysisFactor`'s,
which contained the necessary information on the model create the factor graph that was fitted. The `AnalysisFactor`'s
were created before we composed the `HierachicalFactor` and we pass it separately when composing the factor graph.
"""

factor_graph = af.FactorGraphModel(*analysis_factor_list, hierarchical_factor)

"""
__Search__

We can now create a non-linear search and used it to the fit the factor graph, using its `global_prior_model` property.
"""
laplace = af.LaplaceOptimiser()

factor_graph_result = factor_graph.optimise(
    laplace,
    paths=af.DirectoryPaths(name=f"ep_{signal_to_noise_ratio}"),
    ep_history=af.EPHistory(kl_tol=0.05), max_steps=5
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
# print(f"Hierarchical Mean Mean = {mean_field.mean[hierarchical_factor.mean]}")
# print(f"Hierarchical Mean Variance = {mean_field.variance[hierarchical_factor.mean]}")
# print(f"Hierarchical Mean 1 Sigma = {np.sqrt(mean_field.variance[hierarchical_factor.mean])}")
# print(f"Hierarchical Mean SD/sqrt(variance) = {mean_field.scale[hierarchical_factor.mean]}")
# print()
#
# print(f"Hierarchical Scatter Mean = {mean_field.mean[hierarchical_factor.sigma]}")
# print(f"Hierarchical Scatter Variance = {mean_field.variance[hierarchical_factor.sigma]}")
# print(f"Hierarchical Scatter 1 Sigma = {np.sqrt(mean_field.variance[hierarchical_factor.sigma])}")
# print(f"Hierarchical Scatter SD/sqrt(variance) = {mean_field.scale[hierarchical_factor.sigma]}")
# print()

mean_variable = list(mean_field.variables)[-2]
scatter_variable = list(mean_field.variables)[-1]

print(f"Hierarchical Mean Mean = {mean_field.mean[mean_variable]}")
print(f"Hierarchical Mean Variance = {mean_field.variance[mean_variable]}")
print(f"Hierarchical Mean 1 Sigma = {np.sqrt(mean_field.variance[mean_variable])}")
print(f"Hierarchical Mean SD/sqrt(variance) = {mean_field.scale[mean_variable]}")
print()

print(f"Hierarchical Scatter Mean = {mean_field.mean[scatter_variable]}")
print(f"Hierarchical Scatter Variance = {mean_field.variance[scatter_variable]}")
print(f"Hierarchical Scatter 1 Sigma = {np.sqrt(mean_field.variance[scatter_variable])}")
print(f"Hierarchical Scatter SD/sqrt(variance) = {mean_field.scale[scatter_variable]}")
print()

"""
Exmaple using original model.
"""
print(f"Normalization Gaussian Dataset 0 Mean = {mean_field.mean[model_list[0].gaussian.normalization]}")


"""
__Wrap Up__

As expected, using a graphical model allows us to infer a more precise and accurate model. Unlike the previous 
tutorial, our model-fit: 

1) Fully exploits the information we know about the global model, for example that the centre of every Gaussian in every 
dataset is aligned. Now, the fit of the Gaussian in dataset 1 informs the fits in datasets 2 and 3, and visa versa.

2) Infers a PDF on the global centre that fully accounts for the degeneracies between the models fitted to different 
datasets. This is not the case when we combine the PDFs of each individual fit.

3) Has a well defined prior on the global centre, instead of 3 independent priors on the centre of each dataset.

The tools introduced in this tutorial form the basis of building graphical models of arbitrary complexity, which may
consist of many hundreds of individual model components with thousands of parameters that are fitted to extremely large
datasets. 

The graphical model fitted in this tutorial requires that there are parameters which are shared across multiple datasets. 
For many model-fitting problems this is not the case and each dataset requires its own unique set of parameters. 

The global model can instead assume these parameters are drawn from the same global parent distribution. 
This is called a hierarchical model and is the topic of the next tutorial.
"""
