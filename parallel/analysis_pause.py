"""
Parallel: Analysis
==================

This script times how long a likelihood evaluation takes when the likelihood functions of `Analysis` classes that are
added together are parallelized.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import logging
import os
import time
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "searches"))

import autofit as af

logging.basicConfig(
    level=logging.INFO
)


def main():
    """
    __Paths__
    """

    class Analysis(af.Analysis):
        def __init__(self):

            super().__init__()


        def log_likelihood_function(self, instance):

            time.sleep(0.48/8)

            return 1.0

    total_analyses = 8

    analysis_list = [
        Analysis() for i in range(total_analyses)
    ]

    analysis = sum(analysis_list)

    """
    __Time__
    
    Time the analysis LH functions both individually and in parallel.
    """
    repeats = 2

    start = time.time()
    for repeat in range(repeats):
        [
            analysis_list[i].log_likelihood_function(instance=None)
            for i in range(total_analyses)
        ]
    lh_time_serial = (time.time() - start) / repeats
    print(
        f"Time To Evaluate LH of {total_analyses} Analysis objects in serial = {lh_time_serial} \n"
    )

    analysis.n_cores = 8

    start = time.time()
    for repeat in range(repeats):
        analysis.log_likelihood_function(instance=None)
    lh_time_parallel = (time.time() - start) / repeats
    print(
        f"Time To Evaluate LH of {total_analyses} Analysis objects in parallel = {lh_time_parallel} \n"
    )

    print(f"Speed up in parallel = {lh_time_serial / lh_time_parallel}")

    """
    Finished.
    """


if __name__ == "__main__":
    main()