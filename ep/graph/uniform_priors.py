"""
Tutorial 4: Expectation Propagation
===================================
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af

"""
__Example Source Code (`af.ex`)__

__Dataset__
"""
total_gaussians = 5

dataset_name_list = []
data_list = []
noise_map_list = []

for dataset_index in range(total_gaussians):

    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__low_snr", dataset_name
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
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    analysis_list.append(analysis)

"""
__Model__
"""
centre_shared_prior = af.UniformPrior(lower_limit=0.1, upper_limit=100.0)

model_list = []

for model_index in range(len(data_list)):

    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!
    gaussian.normalization = af.UniformPrior(lower_limit=0.1, upper_limit=1e2)
    gaussian.sigma = af.UniformPrior(lower_limit=0.1, upper_limit=25.0)

    model_list.append(gaussian)

"""
__Analysis Factors__

Now we have our `Analysis` classes and graphical model, we can compose our `AnalysisFactor`'s.

However, unlike the previous tutorials, each `AnalysisFactor` is now assigned its own `search`. This is because the EP 
framework performs a model-fit to each node on the factor graph (e.g. each `AnalysisFactor`). Therefore, each node 
requires its own non-linear search, and in this tutorial we use `dynesty`. For complex graphs consisting of many 
nodes, one could easily use different searches for different nodes on the factor graph.

Each `AnalysisFactor` is also given a `name`, corresponding to the name of the dataset it fits. These names are used
to name the folders containing the results in the output directory.
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
__Factor Graph__

"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
__Expectation Propagation__
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(
    name=path.join(
        "ep", "graph", "uniform_priors"
    )
)

factor_graph_result = factor_graph.optimise(
    optimiser=laplace, paths=paths, ep_history=af.EPHistory(kl_tol=0.05)
)


"""
__Output__

The results of the factor graph, using the EP framework and message passing, are contained in the folder 
`output/howtofit/chapter_graphical_models/tutorial_3_expectation_propagation`. 

The following folders and files are worth of note:

 - `graph.info`: this provides an overall summary of the graphical model that is fitted, including every parameter, 
 how parameters are shared across `AnalysisFactor`'s and the priors associated to each individual parameter.
 
 - The 3 folders titled `gaussian_x1_#__low_snr` correspond to the three `AnalysisFactor`'s and therefore signify 
 repeated non-linear searches that are performed to fit each dataset.
 
 - Inside each of these folders are `optimization_#` folders, corresponding to each model-fit performed over cycles of
 the EP fit. A careful inspection of the `model.info` files inside each folder reveals how the priors are updated
 over each cycle, whereas the `model.results` file should indicate the improved estimate of model parameters over each
 cycle.
"""

print(factor_graph_result)

print(factor_graph_result.updated_ep_mean_field.mean_field)

"""
__Output__

The MeanField object representing the posterior.
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
