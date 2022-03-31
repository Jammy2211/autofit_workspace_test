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
    dataset_name = "gaussian_x1__100000_pixels"
    path_prefix = path.join("parallel")

    """
    __Data__
    
    This example fits a single 1D Gaussian, we therefore load and plot data containing one Gaussian.
    """
    dataset_path = path.join("dataset", "example_1d", dataset_name)
    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    """
    __Model + Analysis__
    
    We create the model and analysis, which in this example is a single `Gaussian` and therefore has dimensionality N=3.
    """
    model = af.Model(af.ex.Gaussian)

    model.centre = af.UniformPrior(lower_limit=490000., upper_limit=500000.)
    model.normalization = af.UniformPrior(lower_limit=24.0, upper_limit=26.0)
    model.sigma = af.UniformPrior(lower_limit=9.0, upper_limit=11.0)

    instance = model.instance_from_prior_medians()

    total_analyses = 8

    analysis_list = [
        af.ex.Analysis(data=data, noise_map=noise_map) for i in range(total_analyses)
    ]

    analysis = sum(analysis_list)

    """
    __Time__
    
    Time the analysis LH functions both individually and in parallel.
    """
    repeats = 10

    start = time.time()
    for repeat in range(repeats):
        [
            analysis_list[i].log_likelihood_function(instance=instance)
            for i in range(total_analyses)
        ]
    lh_time_serial = (time.time() - start) / repeats
    print(
        f"Time To Evaluate LH of {total_analyses} Analysis objects in serial = {lh_time_serial} \n"
    )

    # This will use multiprocessing, should be 8 times faster.

    analysis.n_cores = 8

    start = time.time()
    for repeat in range(repeats):
        analysis.log_likelihood_function(
            instance=instance
        )
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