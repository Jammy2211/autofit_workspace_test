"""
Searches: Nautilus
=======================

This example illustrates how to use the nested sampling algorithm Nautilus.

Information about Dynesty can be found at the following links:

 - https://github.com/joshspeagle/dynesty
 - https://dynesty.readthedocs.io/en/latest/
"""
import numpy as np
from os import path
import os
import time

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "searches"))

conf.instance["general"]["model"]["ignore_prior_limits"] = True

import autofit as af
from autofit import exc

def prior_transform(cube, model):

    # `vector_from_unit_vector has a bug which is why we return cube, need to fix.

    return cube

    return model.vector_from_unit_vector(
        unit_vector=cube,
        ignore_prior_limits=True
    )


class Fitness:

    def __init__(
            self, analysis, model,
    ):

        self.analysis = analysis
        self.model = model

    def __call__(self, parameters, *kwargs):

        figure_of_merit = self.figure_of_merit_from(parameter_list=parameters)

        try:
            figure_of_merit = self.figure_of_merit_from(parameter_list=parameters)

            if np.isnan(figure_of_merit):
                return self.resample_figure_of_merit

            return figure_of_merit

        except exc.FitException:
            return self.resample_figure_of_merit

    def fit_instance(self, instance):
        log_likelihood = self.analysis.log_likelihood_function(instance=instance)

        return log_likelihood

    def log_likelihood_from(self, parameter_list):
        instance = self.model.instance_from_vector(vector=parameter_list)
        log_likelihood = self.fit_instance(instance)

        return log_likelihood

    def figure_of_merit_from(self, parameter_list):
        """
        The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space.

        All Nested samplers use the log likelihood.
        """
        return self.log_likelihood_from(parameter_list=parameter_list)

    @property
    def resample_figure_of_merit(self):
        """
        If a sample raises a FitException, this value is returned to signify that the point requires resampling or
        should be given a likelihood so low that it is discard.

        -np.inf is an invalid sample value for Nautilus, so we instead use a large negative number.
        """
        print("RESAMPLED TO NEGATIVE")
        return -1.0e99

def fit():

    """
    __Data__

    This example fits a single 1D Gaussian, we therefore load and plot data containing one Gaussian.
    """
    dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    """
    __Model + Analysis__
    
    We create the model and analysis, which in this example is a single `Gaussian` and therefore has dimensionality N=3.
    """
    model = af.Model(af.ex.Gaussian)

    model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
    model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    fitness = Fitness(
        model=model,
        analysis=analysis,
    )

    n_live = 16
    n_dim = model.prior_count

    points = np.zeros((n_live, n_dim))

    for i in range(n_live):

        point = model.random_vector_from_priors_within_limits()
        points[i, :] = point

    from nautilus import Sampler

    number_of_cores_list = [1, 2, 4, 8, 16]

    for number_of_cores in number_of_cores_list:

        sampler = Sampler(
            prior=prior_transform,
            likelihood=fitness.__call__,
            n_dim=model.prior_count,
            prior_kwargs={"model": model},
            pool=number_of_cores,
            n_live=n_live,
        )

        transform = sampler.prior

        args = list(map(transform, np.copy(points)))

        start = time.time()

        if number_of_cores > 1:
            list(sampler.pool_l.map(sampler.likelihood, args))
        else:
            list(map(sampler.likelihood, args))
            time_lh_x1 = time.time() - start

        time_lh = time.time() - start

        print(f"N_CPU: {number_of_cores} / LH Time: {time_lh} / Speed up: {time_lh_x1 / time_lh}")

if __name__ == "__main__":

    fit()