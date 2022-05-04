"""
EP: Gaussian Priors
===================
"""
from os import path

import autofit as af

"""
__Dataset__
"""
total_datasets = 3

data_list = []
noise_map_list = []

for dataset_index in range(total_datasets):

    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__hierarchical", dataset_name
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    data_list.append(data)
    noise_map_list.append(noise_map)

"""
__Analysis__

For each dataset we now create a corresponding `Analysis` class, like in the previous tutorial.
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)
    analysis_list.append(analysis)

"""
__Model__
"""
model_list = []

for model_index in range(len(data_list)):

    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = af.UniformPrior(
        lower_limit=0.0, upper_limit=100.0
    )
    gaussian.normalization = af.UniformPrior(lower_limit=1.0, upper_limit=10.0)
    gaussian.sigma = af.UniformPrior(lower_limit=5.0, upper_limit=15.0)

    model_list.append(gaussian)

"""
__Analysis Factors__
"""
dynesty = af.DynestyStatic(nlive=100, sample="rwalk")

analysis_factor_list = []

dataset_index = 0

for model, analysis in zip(model_list, analysis_list):

    dataset_name = f"dataset_{dataset_index}"
    dataset_index += 1

    analysis_factor = af.AnalysisFactor(
        prior_model=model, analysis=analysis, optimiser=dynesty, name=dataset_name
    )

    analysis_factor_list.append(analysis_factor)

"""
__Model__

We now compose the hierarchical model that we fit, using the individual Gaussian model components we created above.

We first create a `HierarchicalFactor`, which represents the parent Gaussian distribution from which we will assume 
that the `centre` of each individual `Gaussian` dataset is drawn. 

For this parent `Gaussian`, we have to place priors on its `mean` and `sigma`, given that they are parameters in our
model we are ultimately fitting for.
"""

hierarchical_factor = af.HierarchicalFactor(
    af.GaussianPrior,
    mean=af.GaussianPrior(mean=50.0, sigma=10, lower_limit=0.0, upper_limit=100.0),
    sigma=af.GaussianPrior(mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0),
)

"""
We now add each of the individual model `Gaussian`'s `centre` parameters to the `hierarchical_factor`.

This composes the hierarchical model whereby the individual `centre` of every `Gaussian` in our dataset is now assumed 
to be drawn from a shared parent distribution. It is the `mean` and `sigma` of this distribution we are hoping to 
estimate.
"""

for model in model_list:

    hierarchical_factor.add_drawn_variable(model.centre)

"""
__Factor Graph__
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list, hierarchical_factor)

"""
__Expectation Propagation__
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(
    name=path.join(
        "ep", "hierarchical", "factor_gaussian_bounded", "uniform_priors"
    )
)

factor_graph_result = factor_graph.optimise(
    optimiser=laplace, paths=paths, ep_history=af.EPHistory(kl_tol=0.05)
)

"""
__Output__
"""

print(factor_graph_result)

print(factor_graph_result.updated_ep_mean_field.mean_field)

"""
__Output__
"""
print(factor_graph_result.updated_ep_mean_field.mean_field)
print()

print(factor_graph_result.updated_ep_mean_field.mean_field.variables)
print()

"""
The logpdf of the posterior at the point specified by the dictionary values
"""
# factor_graph_result.updated_ep_mean_field.mean_field(values=None)
print()

"""
A dictionary of the mean with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.mean)
print()

"""
A dictionary of the variance with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.variance)
print()

"""
A dictionary of the s.d./variance**0.5 with variables as keys.
"""
print(factor_graph_result.updated_ep_mean_field.mean_field.scale)
print()

"""
self.updated_ep_mean_field.mean_field[v: Variable] gives the Message/approximation of the posterior for an individual variable of the model
"""
# factor_graph_result.updated_ep_mean_field.mean_field["help"]

"""
Finish.
"""
