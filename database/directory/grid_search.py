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
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.
"""
session = None

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
The code below loads the dataset and sets up the Analysis class.
"""
dataset_path = path.join("dataset", "example_1d", dataset_name)

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

search = af.DynestyStatic(
    path_prefix=path.join("database", "directory"),
    name="extra_search",
    unique_tag=dataset_name,
    session=session,
)

search.fit(model=model, analysis=analysis)

"""
Resultsare written directly to the `database.sqlite` file omitted hard-disc output entirely, which
can be important for performing large model-fitting tasks on high performance computing facilities where there
may be limits on the number of files allowed. The commented out code below shows how one would perform
direct output to the `.sqlite` file. 
"""
search = af.DynestyStatic(
    name="grid_search",
    path_prefix=path.join("database", "directory"),
    number_of_cores=4,
    unique_tag=dataset_name,
    session=session,
)

parent = af.DynestyStatic(name="parent")

grid_search = af.SearchGridSearch(search=dynesty, number_of_steps=2, number_of_cores=2)

grid_search_result = grid_search.fit(
    model=model, analysis=analysis, grid_priors=[model.gaussian.centre], parent=parent
)

"""
Scrape directory to create .sqlite file.
"""
import os
import time
from autofit.database.aggregator import Aggregator

database_file = "database_directory_grid_search.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

agg = Aggregator.from_database(database_file, completed_only=False)

start = time.time()
agg.add_directory(directory=path.join("output", "database", "directory", dataset_name))
print(f"Time to add directory to database {time.time() - start}")


"""
Make sure database + agg can be used.
"""
samples_gen = agg.values("samples")

"""
When we convert this generator to a list and it, the outputs are 3 different SamplesMCMC instances. These correspond to 
the 3 model-fits performed above.
"""
start = time.time()
gaussian = agg.model.gaussian
agg_query = agg.query(gaussian == af.ex.Gaussian)
print("Total queries for correct model = ", len(agg_query))
print(f"Time to query based on correct model {time.time() - start} \n")

name = agg.search.name
agg_query = agg.query(name == "database_grid_search")
print("Total Samples Objects via `name` model query = ", len(agg_query), "\n")

"""
Test that we can retrieve an aggregator with only the grid search results:
"""
agg_grid_searches = agg.grid_searches()
print("Total aggregator via `grid_searches` query = ", len(agg_grid_searches), "\n")
unique_tag = agg_grid_searches.search.unique_tag
agg_qrid = agg_grid_searches.query(unique_tag == "gaussian_x1")

print(
    "Total aggregator via `grid_searches` & unique tag query = ",
    len(agg_grid_searches),
    "\n",
)

"""
The `GridSearchResult` is accessible via the database.
"""
grid_search_result = list(agg_grid_searches)[0]["result"]
print(grid_search_result.best_result)
print(grid_search_result.log_evidences_native)

"""
From the GridSearch, get an aggregator which contains only the maximum log likelihood model. E.g. if the 10th out of the 
16 cells was the best fit:
"""
agg_best_fit = agg_grid_searches.best_fits()
print("Size of Agg best fit = ", len(agg_best_fit), "\n")
instance = agg_best_fit.values("instance")[0]
print(instance.gaussian.sigma)
samples = agg_best_fit.values("samples")[0]
print(samples)

"""
Reqest 3:

From the GridSearch, get an aggregator for any of the grid cells.
"""
cell_aggregator = agg_grid_searches.cell_number(1)
print(cell_aggregator)
print("Size of Agg cell = ", len(cell_aggregator), "\n")

"""
Stored and prints input parent grid of grid search.
"""

for fit in agg_best_fit:
    print(fit.parent)
