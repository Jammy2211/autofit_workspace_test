"""
Feature: Database
=================

The default behaviour of **PyAutoFit** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check. For small model-fitting tasks this is sufficient, however many users
have a need to perform many model fits to very large datasets, making manual inspection of results time consuming.

PyAutoFit's database feature outputs all model-fitting results as a
sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database, such that all results
can be efficiently loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation. This
database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can be
loaded.

This example extends our example of fitting a 1D `Gaussian` profile and fits 3 independent datasets each containing a
1D Gaussian. The results will be written to a `.sqlite` database, which we will load to demonstrate the database.

A full description of PyAutoFit's database tools is provided in the database chapter of the `HowToFit` lectures.
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

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
Results are written directly to the `database.sqlite` file omitted hard-disc output entirely, which
can be important for performing large model-fitting tasks on high performance computing facilities where there
may be limits on the number of files allowed. The commented out code below shows how one would perform
direct output to the `.sqlite` file. 
"""

search = af.DynestyStatic(
    name="pickle_remove",
    path_prefix=path.join("database", "directory"),
    nlive=75,
    number_of_cores=1,
    unique_tag=dataset_name,
    session=session,
)

result = search.fit(model=model, analysis=analysis, info={"example_key": 42})

"""
__Database__

The results are not contained in the `output` folder after each search completes. Instead, they are
contained in the `database.sqlite` file, which we can load using the `Aggregator`.
"""
from autofit.database.aggregator import Aggregator

database_file = "database_directory_pickle_remove.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass


agg = Aggregator.from_database(path.join(database_file))
agg.add_directory(
    directory=path.join("output", "database", "directory", dataset_name, "pickle_remove")
)

"""
__Search__

Load the search via the database and make assert statements to ensure that using `search.json` works as expected.
"""
search = agg.values(name="search")[0]

print(search.paths.name)

assert search.paths.name == "pickle_remove"
assert search.nlive == 75

"""
__Model__

Load the `model` via the database and put some assert statements to ensure that using `model.json` works as expected.
"""
model = agg.values(name="model")[0]

assert model.gaussian.centre.lower_limit == 0.0
assert model.gaussian.centre.upper_limit == 100.0
assert model.gaussian.normalization.lower_limit == 1e-2


"""
__Info__

Load the `info` via the database and put some assert statements to ensure that using `info.json` works as expected.
"""
info = agg.values(name="info")[0]

assert info["example_key"] == 42

"""
__Samples__

Load the `Samples` via the database and put some assert statements to ensure that using `samples.csv` works as
expected.
"""
samples = agg.values(name="samples")[0]

max_lh_instance = samples.max_log_likelihood()

assert max(samples.log_likelihood_list) > -46.0
assert 47 < max_lh_instance.gaussian.centre < 53
assert samples.log_evidence > -60.0

"""
__Results Internal__

Using the `Samples` loaded above, verify that the results internal have been loaded via a .pickle file and thus
dynesty visualization works as expected.
"""
import autofit.plot as aplt

search_plotter = aplt.DynestyPlotter(samples=samples)

search_plotter.cornerplot()