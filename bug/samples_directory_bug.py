"""
Feature: Database
=================

Tests that general results can be loaded from hard-disk via directory aggregation.

This script outputs all files which can be associated with a model-fit (e.g. samples, full samples summary, search
output). This can take up large amounts of hard-disk space.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af

import os
from os import path
import numpy as np

"""
__Dataset Names__

Load the dataset from hard-disc, set up its `Analysis` class and fit it with a non-linear search. 
"""
dataset_name = "gaussian_x1"

"""
__Model__

Next, we create our model, which again corresponds to a single `Gaussian` with manual priors.
"""
model = af.Collection(gaussian=af.ex.Gaussian)

model.gaussian.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.gaussian.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.gaussian.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.
"""
session = None

"""
The code below loads the dataset and sets up the Analysis class.
"""
dataset_path = path.join("dataset", "example_1d", dataset_name)

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
Default example Analysis does not output a .pickle file and does not test pickle loading.

We extend the Analysis class to output the data as a pickle file, which we test can be loaded below
"""

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
Results are written directly to the `database.sqlite` file omitted hard-disc output entirely, which
can be important for performing large model-fitting tasks on high performance computing facilities where there
may be limits on the number of files allowed. The commented out code below shows how one would perform
direct output to the `.sqlite` file. 
"""
name = "general"

search = af.DynestyStatic(
    name=name,
    path_prefix=path.join("database", "directory"),
    number_of_cores=1,
    unique_tag=dataset_name,
    session=session,
    maxcall=100,
    maxiter=100,
)

result = search.fit(model=model, analysis=analysis)

"""
__Database__
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "database", "directory", dataset_name, name),
)

"""
__Samples + Results__

Make sure database + agg can be used.
"""
for samples in agg.values("samples"):
    print(type(samples))

    seojmfposd
