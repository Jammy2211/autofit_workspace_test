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
agg.values("model")
print(f"Time to load model via pickle {time.time() - start} \n")

start = time.time()
agg.values("samples")
print(f"Time to load samples via pickle {time.time() - start} \n")
