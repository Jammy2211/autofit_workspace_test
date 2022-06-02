"""
Fit every 1D Gaussian with a shared centre simultaneously, with the centre parameter identical across all
datasets.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

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
centre_shared_prior = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)

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

    gaussian.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)
    gaussian.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
    gaussian.centre = centre_shared_prior  # This prior is used by all 3 Gaussians!

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
analysis_factor_list = []

for model, analysis in zip(model_list, analysis_list):

    analysis_factor = af.AnalysisFactor(prior_model=model, analysis=analysis)

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
search = af.DynestyStatic(
    name=f"simultaneous_{signal_to_noise_ratio}",
    nlive=300,
    dlogz=1e-4,
    sample="rwalk",
    walks=10,
)

result = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)

"""
__Result__

We can now inspect the inferred value of `centre`, and compare this to the value we estimated in the previous tutorial
via a joint PDF. 

If the graphical model was successful, we should see the errors dramatically reduce!

(The errors of the joint PDF below is what was estimated for a run on my PC, yours may be slightly different!)
"""

centre = result.samples.median_pdf_instance[0].gaussian.centre

u1_error = result.samples.instance_at_upper_sigma(sigma=1.0)[0].gaussian.centre
l1_error = result.samples.instance_at_lower_sigma(sigma=1.0)[0].gaussian.centre

u3_error = result.samples.instance_at_upper_sigma(sigma=3.0)[0].gaussian.centre
l3_error = result.samples.instance_at_lower_sigma(sigma=3.0)[0].gaussian.centre

print(f"Inferred value of the shared centre via a Joint PDF of fits to {total_datasets} datasets: \n")
print(f"{centre} ({l1_error} {u1_error}) ({u1_error - l1_error}) [1.0 sigma confidence intervals]")
print(f"{centre} ({l3_error} {u3_error}) ({u3_error - l3_error}) [3.0 sigma confidence intervals]")

samples = result.samples

output = aplt.Output(
    path=plot_path, filename=f"centre_{signal_to_noise_ratio}", format="png"
)

"""
__Likelihood Check__
"""
print(f"Overall Likelihood = {max(samples.log_likelihood_list)}")
print()

# """
# __PDFs__
# """
# from getdist import MCSamples
# from getdist import plots
# import matplotlib.pyplot as plt
# import numpy as np
#
# gd_samples = MCSamples(
#     samples=np.asarray(samples.parameter_lists),
#     loglikes=np.asarray(samples.log_likelihood_list),
#     weights=np.asarray(samples.weight_list),
#     names=samples.model.parameter_names,
#     labels=samples.model.parameter_labels_with_superscripts_latex,
#     sampler="nested",
# )
#
# gd_plotter = plots.get_subplot_plotter(width_inch=12)
#
# gd_plotter.triangle_plot(roots=gd_samples, filled=False)
#
# plt.savefig(path.join(plot_path, f"pdf_{signal_to_noise_ratio}_{total_datasets}.png"))
# plt.close()


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
