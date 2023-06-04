"""
__Example: Analysis Summing__

In this example, we'll sum together `Analysis` classes to fit multiple datasets with the same model.

We'll illustrate analysis class summing with two different examples:

  - Instances of the same `Analysis` class are summed together which contain different datasets with the same 1D
  Gaussian signal, but each with a different noise-map realization. Summing each instance of the analysis therefore
  allows us to infer more precise model parameters, as multiple datasets are fitted by the same non-linear search.

  - Instances of different `Analysis` classes are summed, where each analysis fit datasets that are different in their
  format and constrain different and unique aspects of the model being fitted. This allows us to constrain a model by
  simultaneously fitting different datasets, something we've not been able to do previously!

If you haven't already, you should checkout the files `example/model.py` and `example/analysis.py` to see how we have
provided PyAutoFit with the necessary information on our model, data and log likelihood function.
"""
import autofit as af
import autofit.plot as aplt

import matplotlib.pyplot as plt
import numpy as np
from os import path

"""
__Data__

First, lets load 3 datasets of a 1D Gaussian, by loading them from .json files in the directory 
`autofit_workspace/dataset/`.

All 3 datasets contain the same Gaussian (e.g. with identical values for `centre`, `normalization` and `intensity`) but
each has a different noise realization.
"""
dataset_size = 3

data_list = []
noise_map_list = []

for dataset_index in range(dataset_size):
    dataset_path = path.join(
        "dataset", "example_1d", f"gaussian_x1_identical_{dataset_index}"
    )

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    data_list.append(data)

    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )
    noise_map_list.append(noise_map)

"""
Now lets plot all 3 datasets, including their error bars. 
"""
for data in data_list:
    xvalues = range(data.shape[0])

    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.show()
    plt.close()

"""
__Model__

Next, we create our model, which in this case corresponds to a single Gaussian. In model.py, you will have noted
this `Gaussian` has 3 parameters (centre, normalization and sigma). These are the free parameters of our model that the
non-linear search fits for, meaning the non-linear parameter space has dimensionality = 3.
"""
model = af.Model(af.ex.Gaussian)

"""
Checkout `autofit_workspace/config/priors/model.json`, this config file defines the default priors of the `Gaussian` 
model component. 

We can overwrite priors before running the `NonLinearSearch` as shown below.
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
__Analysis__

We now set up our three instances of the Analysis class, using the class described in `analysis.py`. As seen in other
examples, we set up an `Analysis` for each dataset one-by-one, using a for loop:
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)
    analysis_list.append(analysis)

"""
__Analysis Summing__

We can now sum together every analysis in the list, to produce an overall analysis class which we fit with a non-linear
search.

Summing analysis objects means the following happens:

 - The output path structure of the results goes to a single folder, which includes sub-folders for visualization
 of every individual analysis object.
 
 - The log likelihood values computed by the `log_likelihood_function` of each individual analsys class are summed to
 give the overall log likelihood value that the non-linear search uses for model-fitting.
"""
analysis = analysis_list[0] + analysis_list[1] + analysis_list[2]

"""
__Search__

We now perform the search using this analysis class, which follows the same API as other tutorials on the workspace.
"""
dynesty = af.DynestyStatic(path_prefix=path.join("misc"), name="analysis_summing_0")

result = dynesty.fit(model=model, analysis=analysis)

"""
__Different Analysis Objects__

The second part of this example which sums together different `Analysis` objects is not written yet. The aim of this
example is to show how even if you have different datasets which are fitted by the model in different ways (e.g. each
with their own `log_likelihood_function` you can use Analysis class summing to easily define a unique Analysis class
for each dataset.

This functionality works in PyAutoFit and can easily be adopted by following the same API shown above, but using your
own Analysis classes.
"""
