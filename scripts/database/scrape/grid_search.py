"""
Feature: Database
=================

Tests that the results of a grid search of searches can be loaded from hard-disk via a database built via a scrape.
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

name = "grid_search"

search = af.DynestyStatic(
    path_prefix=path.join("database", "scrape", name),
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
    name=name,
    path_prefix=path.join("database", "scrape", name),
    number_of_cores=1,
    unique_tag=dataset_name,
    session=session,
    force_x1_cpu=True,  # ensures parallelizing over grid search works.
)

parent = af.DynestyStatic(name="parent")

grid_search = af.SearchGridSearch(search=search, number_of_steps=2, number_of_cores=1)

grid_search_result = grid_search.fit(
    model=model, analysis=analysis, grid_priors=[model.gaussian.centre], parent=parent
)

"""
Scrape directory to create .sqlite file.
"""
import os
from autofit.database.aggregator import Aggregator

database_file = "database_directory_grid_search.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

agg = Aggregator.from_database(database_file, completed_only=False)

agg.add_directory(
    directory=path.join("output", "database", "scrape", name, dataset_name)
)

assert len(agg) > 0

"""
Make sure database + agg can be used.
"""
print("\n\n***********************")
print("****RESULTS TESTING****")
print("***********************\n")

for samples in agg.values("samples"):
    print(samples.parameter_lists[0])

mp_instances = [samps.median_pdf() for samps in agg.values("samples")]
print(mp_instances)

"""
__Queries__
"""
print("\n\n***********************")
print("****QUERIES TESTING****")
print("***********************\n")

gaussian = agg.model.gaussian
agg_query = agg.query(gaussian == af.ex.Gaussian)
print("Total queries for correct model = ", len(agg_query))

name = agg.search.name
agg_query = agg.query(name == "database_grid_search")
print("Total Samples Objects via `name` model query = ", len(agg_query), "\n")

"""
__Grid Search Results__
"""
print("\n\n***********************")
print("**GRID RESULTS TESTING**")
print("***********************\n\n")

agg_grid_searches = agg.grid_searches()
print("\n****Total aggregator via `grid_searches` query = ", len(agg_grid_searches), "****\n")
unique_tag = agg_grid_searches.search.unique_tag
agg_qrid = agg_grid_searches.query(unique_tag == "gaussian_x1")

print(
    "****Total aggregator via `grid_searches` & unique tag query = ",
    len(agg_grid_searches),
    "****\n",
)

"""
The `GridSearchResult` is accessible via the database.
"""
grid_search_result = list(agg_grid_searches)[0]["result"]
print(f"****Best result (grid_search_result.best_result)****\n\n {grid_search_result.best_result}\n")
print(f"****Grid Log Evidences (grid_search_result.log_evidences_native)****\n\n {grid_search_result.log_evidences_native}\n")

"""
From the GridSearch, get an aggregator which contains only the maximum log likelihood model. E.g. if the 10th out of the 
16 cells was the best fit:
"""
print("\n\n****MAX LH AGGREGATOR VIA GRID****\n\n")

agg_best_fit = agg_grid_searches.best_fits()
print(f"Max LH Gaussian sigma (agg_best_fit.values('instance')[0].gaussian.sigma) {agg_best_fit.values('instance')[0]}\n")
print(f"Max LH samples (agg_best_fit.values('samples')[0]) {agg_best_fit.values('samples')[0]}")


"""
From the GridSearch, get an aggregator for any of the grid cells.
"""
print("\n\n****AGGREGATOR FROM INPUT GRID CELL****\n\n")

cell_aggregator = agg_grid_searches.cell_number(1)
print(f"Cell Aggregator (agg_grid_searches.cell_number(1)) {cell_aggregator}")
print("Size of Agg cell = ", len(cell_aggregator), "\n")

"""
Stored and prints input parent grid of grid search.
"""

for fit in agg_best_fit:
    print(f"Grid Search Parent (fit.parent): {fit.parent}")