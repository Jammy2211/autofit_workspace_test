"""
Multiple Datasets
=================
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autofit as af
import autofit.plot as aplt

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
        "dataset", "example_1d", f"gaussian_x1_multi", f"dataset_{dataset_index}"
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
dynesty = af.DynestyStatic(path_prefix="multi", name="x1_parameter")

result_list = dynesty.fit(model=model, analysis=analysis)

"""
__Result__
"""
model_data = result_list[0].max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)
