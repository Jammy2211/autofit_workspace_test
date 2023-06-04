"""
Feature: Database
=================

This test is the same as `general.py` but with an analysis objected that is a `CombinedAnalysis` of 3 identical
`Analysis` objects.

This tests whether database functionality works for multi-dataset fitting.
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

for i in range(3):
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

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    """
    We now create a list of 3 identical analysis objects and sum them, to activate PyAutoFit's multi-dataset
    fitting feature.
    """
    analysis_list = [analysis, analysis, analysis]

    analysis = sum(analysis_list)

    """
    Results are written directly to the `database.sqlite` file omitted hard-disc output entirely, which
    can be important for performing large model-fitting tasks on high performance computing facilities where there
    may be limits on the number of files allowed. The commented out code below shows how one would perform
    direct output to the `.sqlite` file. 
    """
    dynesty = af.DynestyStatic(
        path_prefix=path.join("database"),
        name="multi_dataset",
        number_of_cores=1,
        unique_tag=f"{dataset_name}_{i}",
        session=session,
    )

    result = dynesty.fit(model=model, analysis=analysis)

"""
__Database__

The results are not contained in the `output` folder after each search completes. Instead, they are
contained in the `database.sqlite` file, which we can load using the `Aggregator`.
"""
from autofit.database.aggregator import Aggregator

database_file = "database.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass


agg = Aggregator.from_database(
    path.join(database_file), completed_only=False, top_level_only=False
)
agg.add_directory(directory=path.join("output", "database"))

"""
__Samples + Results__

Make sure database + agg can be used.
"""
for samples in agg.values("samples"):
    print(samples.parameter_lists[0])

mp_instances = [samps.median_pdf() for samps in agg.values("samples")]
print(mp_instances)

"""
__Queries__
"""
path_prefix = agg.search.path_prefix
agg_query = agg.query(path_prefix == path.join("database", "session", dataset_name))
print("Total Samples Objects via `path_prefix` model query = ", len(agg_query), "\n")

name = agg.search.name
agg_query = agg.query(name == "multi_dataset")
print("Total Samples Objects via `name` model query = ", len(agg_query), "\n")

gaussian = agg.model.gaussian
agg_query = agg.query(gaussian == af.ex.Gaussian)
print("Total Samples Objects via `Gaussian` model query = ", len(agg_query), "\n")

gaussian = agg.model.gaussian
agg_query = agg.query(gaussian.sigma > 3.0)
print("Total Samples Objects In Query `gaussian.sigma < 3.0` = ", len(agg_query), "\n")

gaussian = agg.model.gaussian
agg_query = agg.query((gaussian == af.ex.Gaussian) & (gaussian.sigma < 3.0))
print(
    "Total Samples Objects In Query `Gaussian & sigma < 3.0` = ", len(agg_query), "\n"
)

unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "gaussian_x1_1")

print(agg_query.values("samples"))
print("Total Samples Objects via unique tag Query = ", len(agg_query), "\n")

"""
__Data__

Loading data via the aggregator, to ensure it is output by the model-fit in pickle files and loadable.

In the `general.py` example this provides a list of `data` objects for every model-fit performed (in this example a 
list with a single entry for the single model-fit performed).

For a `CombinedAnalysis` this should provide a list of `data` objects for every model-fit performed, but where
every entry in the list is a list of `data` objects corresponding to each `data` used in each individual `Analysis`.

There is an example in the comment below.
"""


def _data_from(fit: af.Fit):
    return fit.child_values(name="data")


data_gen = agg.child_values(name="data")

data = [data for data in data_gen]

print("Data via Data Gen:")
print(data)

# If 5 model-fits are performed using CombinedAnalysis, each with 3 datasets, this should return the second data '
# of the 4th model fit.

print(data[0][2])
