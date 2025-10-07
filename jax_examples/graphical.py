import autofit as af
from autofit.non_linear.fitness import Fitness

from os import path
import jax
from jax import grad

"""
__Dataset Names__
"""
dataset_name = "gaussian_x1"

"""
__Model__
"""
model = af.Collection(gaussian=af.ex.Gaussian)

parameters = model.physical_values_from_prior_medians
instance = model.instance_from_prior_medians()

"""
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.
"""
session = None

"""
__Dataset__
"""
dataset_path = path.join("dataset", "example_1d", dataset_name)

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
__Analysis__
"""
analysis = af.ex.Analysis(
    data=data,
    noise_map=noise_map,
)

"""
__Normal Model__
"""
# fitness = Fitness(
#     model=model,
#     analysis=analysis,
#     fom_is_log_likelihood=True,
#     resample_figure_of_merit=-1.0e99,
# )
#
# grad = jax.jit(grad(fitness.analysis.log_likelihood_function))
#
# print(grad(instance))
#
# fitness = Fitness(
#     model=model,
#     analysis=analysis,
#     fom_is_log_likelihood=True,
#     resample_figure_of_merit=-1.0e99,
# )
#
# grad = jax.jit(grad(fitness))
#
# print(grad(parameters))
#
# ffff


"""
__Factor Graph__
"""
analysis_factor_list = []

for analysis0 in [analysis, analysis]:

    analysis_model = model.copy()

    analysis_factor = af.AnalysisFactor(prior_model=analysis_model, analysis=analysis0)

    analysis_factor_list.append(analysis_factor)

factor_graph = af.FactorGraphModel(*analysis_factor_list)

fitness = Fitness(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

fitness(parameters)
grad = jax.jit(grad(fitness))

print(grad(parameters))
