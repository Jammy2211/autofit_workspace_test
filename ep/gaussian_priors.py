"""
EP: Gaussian Priors
===================
"""
from os import path

import autofit as af

"""
__Dataset__
"""
total_datasets = 5

data_list = []
noise_map_list = []

for dataset_index in range(total_datasets):

    dataset_name = f"dataset_{dataset_index}"

    dataset_path = path.join(
        "dataset", "example_1d", "gaussian_x1__low_snr", dataset_name
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
centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0)

model_list = []

for model_index in range(len(data_list)):

    gaussian = af.Model(af.ex.Gaussian)

    gaussian.centre = centre_shared_prior

    gaussian.normalization = af.GaussianPrior(mean=3.0, sigma=5.0)
    gaussian.sigma = af.GaussianPrior(mean=10.0, sigma=10.0)

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
__Factor Graph__
"""
factor_graph = af.FactorGraphModel(*analysis_factor_list)

"""
__Expectation Propagation__
"""
laplace = af.LaplaceOptimiser()

paths = af.DirectoryPaths(
    name=path.join(
        "ep", "gaussian_priors"
    )
)

factor_graph_result = factor_graph.optimise(
    optimiser=laplace, paths=paths, ep_history=af.EPHistory(kl_tol=0.2)
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
