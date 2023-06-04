"""
Multiple Datasets
=================
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "searches"))

import autofit as af

"""
__Data__

First, lets load 2 datasets of a 1D Gaussian, by loading them from .json files in the directory 
`autofit_workspace/dataset/`.

All 3 datasets contain the same Gaussian except the `sigma` parameter is different, and we will fit a model where
all parameters except `sigma` are shared.
"""
dataset_size = 2

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
__Analysis__

We now set up our three instances of the Analysis class, using the class described in `analysis.py`. As seen in other
examples, we set up an `Analysis` for each dataset one-by-one, using a for loop:
"""
analysis_list = []

for data, noise_map in zip(data_list, noise_map_list):
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)
    analysis_list.append(analysis)

analysis = sum(analysis_list)

"""
__Model__

Next, we create our model, which in this case corresponds to a single Gaussian.
"""
model = af.Model(af.ex.Gaussian)

"""
We now make the `sigma` a free parameter across every analysis object.
"""
analysis = analysis.with_free_parameters(model.sigma)

"""
__Search__

We now perform the search using this analysis class, which follows the same API as other tutorials on the workspace.
"""
search = af.DynestyStatic(path_prefix="multi", name="x1_parameter")

result_list = search.fit(model=model, analysis=analysis)

"""
__Result__
"""
model_data = result_list[0].max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)

"""
__Model Linking 1 (To Instances)__

Compose a model via model linking where some parameters are passed as instances. 

This would traditionally use the model-linking API via something like:

 model = af.Model(af.ex.Gaussian)

 model.sigma = result_list.instance.sigma
 model.normalization = result_list.instance.normalization

Note that there is slightly different behaviour for sigma and normalizaiton above:

 - The sigma parameter was free across every analysis object in the model fitted above, thus each `model.sigma` value 
 should be different based on this result.
 
 - The normalization parameter was the same for all analysis objects, so should be the same value for the model.
"""
# model_link_2 = NEW API

search = af.DynestyStatic(path_prefix="multi", name="model_linking_2")
search.fit(model=model_link_2, analysis=analysis)

"""
__Model Linking 2 (To Model)__

Compose a model via model linking where parameters become model-components treated as `GaussianPriors`. 

This would traditionally use the model-linking API via something like:

 model.sigma = result_list.model.sigma
 model.normalization = result_list.model.normalization

For sigma, it is straight forward to imagine how it is passed. It had free parameters for both analysis objects
in the fit above, thus we should pass it as two separate model components.

We have to make a decision about how the model for normalization is passed:

 - The normaliation values were shared in the previous fit, passing it as a model could retain this information and
   pass it as just 1 free parameter.

 - It could make thee normalization free across both analysis objects, ignoring the parameterization of the fit performed
   previously.
 
I suggest that ``model`` retains the information of the previous analysis, therefore the simpler model is linked.
"""
# model_link_2 = NEW API

search = af.DynestyStatic(path_prefix="multi", name="model_linking_1")
search.fit(model=model_link_1, analysis=analysis)

"""
__Model Linking 3 (Relational Models)__

This may require some dedicated thought...
"""
# model_link_3 = NEW API

search = af.DynestyStatic(path_prefix="multi", name="model_linking_2")
search.fit(model=model_link_3, analysis=analysis)
