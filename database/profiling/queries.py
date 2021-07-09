# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af

from os import path
import numpy as np

import model as m
import analysis as a

import time
import os

"""
__Database__

First, note how the results are not contained in the `output` folder after each search completes. Instead, they are
contained in the `queries_profiling.sqlite` file, which we can load using the `Aggregator`.
"""
agg = af.Aggregator.from_database("profiling.sqlite")

start = time.time()
gaussian = agg.gaussian
agg_query = agg.query(gaussian == m.Gaussian)
print("Total queries for correct model = ", len(agg_query))
print(f"Time to query based on correct model {time.time() - start} \n")

start = time.time()
gaussian = agg.gaussian
agg_query = agg.query(gaussian != m.Gaussian)
print(f"Time to query based on incorrect model {time.time() - start} \n")

# start = time.time()
# agg_query = agg.query(agg.search.name == None)
# print("Total queries for correct name = ", len(agg_query))
# print(f"Time to query based on correct name {time.time() - start} \n")

start = time.time()
agg_query = agg.query(agg.search.name == "wrong")
print(f"Time to query based on incorrect name {time.time() - start} \n")
