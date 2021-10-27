"""
Tutorial 3: Expectation Propagation
===================================

In the previous tutorial, we fitted a graphical model to dataset comprising 3 noisy 1D Gaussians, which had a shared
and global value of their `centre`. This provided a robust estimate of the global value of centre, and provides the
basis of composing and fitting complex models to large datasets.

We concluded by discussing that one would soon hit a ceiling scaling these graphical models up to extremely large
datasets. One would soon find that the parameter space is too complex to sample, and computational limits would
ultimately cap how many datasets one could feasible fit.

This tutorial introduces expectation propagation (EP), the solution to this problem, which inspects a factor graph
and partitions the model-fit into many simpler fits of sub-components of the graph to individual datasets. This
overcomes the challenge of model complexity, and mitigates computational restrictions that may occur if one tries to
fit every dataset individually.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
from os import path

"""
__Dataset__

For each dataset we now set up the correct path and load it. 

[Do we need to introduce a graphical model API for loading data? Just use lists / for loop?] 
"""
dataset_path = path.join("dataset", "example_1d")

dataset_0_path = path.join(dataset_path, "gaussian_x1_0__low_snr")
data_0 = af.util.numpy_array_from_json(file_path=path.join(dataset_0_path, "data.json"))
noise_map_0 = af.util.numpy_array_from_json(
    file_path=path.join(dataset_0_path, "noise_map.json")
)

dataset_1_path = path.join(dataset_path, "gaussian_x1_1__low_snr")
data_1 = af.util.numpy_array_from_json(file_path=path.join(dataset_1_path, "data.json"))
noise_map_1 = af.util.numpy_array_from_json(
    file_path=path.join(dataset_1_path, "noise_map.json")
)

dataset_2_path = path.join(dataset_path, "gaussian_x1_2__low_snr")
data_2 = af.util.numpy_array_from_json(file_path=path.join(dataset_2_path, "data.json"))
noise_map_2 = af.util.numpy_array_from_json(
    file_path=path.join(dataset_2_path, "noise_map.json")
)

"""
__Analysis__

For each dataset we now create a corresponding `Analysis` class, just like we did in the previous tutorial.

[Again, PyAutoFit specific API or just for loops?]
"""
analysis_0 = af.ex.Analysis(data=data_0, noise_map=noise_map_0)
analysis_1 = af.ex.Analysis(data=data_1, noise_map=noise_map_1)
analysis_2 = af.ex.Analysis(data=data_2, noise_map=noise_map_2)

"""
__Model__

We now compose the graphical model that we fit, using the `Model` and `Collection` objects you are now familiar with.
"""
from autofit import graphical as g

"""
We begin by setting up a shared prior for `centre`, just like we did in the previous tutorilal

[Again, what is the PyAutoFit API for this? Hierachical models?]
"""
centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0)

gaussian_0 = af.Model(af.ex.Gaussian)
gaussian_0.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!
gaussian_0.normalization = af.GaussianPrior(mean=3.0, sigma=5.0, lower_limit=0.0)
gaussian_0.sigma = af.GaussianPrior(mean=10.0, sigma=10.0, lower_limit=0.0)

prior_model_0 = af.Collection(gaussian=gaussian_0)

gaussian_1 = af.Model(af.ex.Gaussian)
gaussian_1.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!
gaussian_1.normalization = af.GaussianPrior(mean=3.0, sigma=5.0, lower_limit=0.0)
gaussian_1.sigma = af.GaussianPrior(mean=10.0, sigma=10.0, lower_limit=0.0)

prior_model_1 = af.Collection(gaussian=gaussian_1)

gaussian_2 = af.Model(af.ex.Gaussian)
gaussian_2.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!
gaussian_2.normalization = af.GaussianPrior(mean=3.0, sigma=5.0, lower_limit=0.0)
gaussian_2.sigma = af.GaussianPrior(mean=10.0, sigma=10.0, lower_limit=0.0)

prior_model_2 = af.Collection(gaussian=gaussian_2)

"""
__Analysis Factors__

Now we have our `Analysis` classes and graphical model, we can compose our `AnalysisFactor`'s, just like we did in the
previous tutorial.

However, unlike the previous tutorial, each `AnalysisFactor` is now assigned its own `search`. This is because the EP 
framework performs a model-fit to each node on the factor graph (e.g. each `AnalysisFactor`). Therefore, each node 
requires its own non-linear search, and in this tutorial we use `dynesty`. For complex graphs consisting of many 
nodes, one could easily use different searches for different nodes on the factor graph.

[Again, PyAutoFit API for making many AnalysisFactors. Loop? List?]
"""
dynesty = af.DynestyStatic(
    path_prefix=path.join("graphical"),
    name="expectation_propagation",
    nlive=100,
    sample="rwalk",
)

analysis_factor_0 = g.AnalysisFactor(
    prior_model=prior_model_0, analysis=analysis_0, optimiser=dynesty
)
analysis_factor_1 = g.AnalysisFactor(
    prior_model=prior_model_1, analysis=analysis_1, optimiser=dynesty
)
analysis_factor_2 = g.AnalysisFactor(
    prior_model=prior_model_2, analysis=analysis_2, optimiser=dynesty
)

"""
We combine our `AnalysisFactors` into one, to compose the factor graph.
"""
factor_graph = g.FactorGraphModel(
    analysis_factor_0, analysis_factor_1, analysis_factor_2
)

"""
__Expectation Propagation__

In the previous tutorial, we used the `global_prior_model` of the `factor_graph` to fit the global model. In this 
tutorial, we instead fit the `factor_graph` using the EP framework, which. fits the graphical model composed in this 
tutorial as follows:

1) Go to the first node on the factor graph (e.g. `analysis_factor_0`) and fit its model to its dataset. This is simply
a fit of the `Gaussian` model to the first 1D Gaussian dataset, the model-fit we are used to performing by now.

2) Once the model-fit is complete, inspect the model for parameters that are shared with other nodes on the factor
graph. In this example, the `centre` of the `Gaussian` fitted to the first dataset is global, and therefore connects
to two other nodes on the factor graph (the `AnalysisFactor`'s) of the second and first `Gaussian` datasets.

3) The EP framework now creates a 'message' that is to be passed to the connecting nodes on the factor graph. This
message informs them of the results of the model-fit, so they can update their priors on the `Gaussian`'s centre 
accordingly and, more importantly, update their posterior inference and therefore estimate of the global centre.

In this tutorial, the EP 
framework first fits all three nodes of the factor graph one by one (using the `dynesty` searches we set up above). 
After each fit, it creates a 'message', that is passed to other nodes on the factor graph to inform them of the results
of the model fit. 

For example, the model fitted to the first Gaussian dataset includes the global centre. Therefore, after the model is 
fitted, the EP framework creates a 'message' 
informs the factor graph 

A crucial part of EP is message passing. 
"""
from autofit.graphical import optimise

laplace = optimise.LaplaceFactorOptimiser()
collection = factor_graph.optimise(laplace)

print(collection)

"""
Finish.
"""
