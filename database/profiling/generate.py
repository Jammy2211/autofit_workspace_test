"""
This script generates a large set of results in a .sqlite database, for profiling the database.

"""
import autofit as af

import os
import time
from os import path
import numpy as np

import model as m
import analysis as a


def simulate_line_from_gaussian(instance):

    """
    Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated using and
    thus defining the number of data-points in our data.
    """
    pixels = 100
    xvalues = np.arange(pixels)

    """
    Evaluate this `Gaussian` model instance at every xvalues to create its model profile.
    """
    try:
        model_line = sum(
            [line.profile_from_xvalues(xvalues=xvalues) for line in instance]
        )
    except TypeError:
        model_line = instance.profile_from_xvalues(xvalues=xvalues)

    """Determine the noise (at a specified signal to noise level) in every pixel of our model profile."""
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

    """
    Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio to compute
    noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
    """
    data = model_line + noise
    noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

    return data, noise_map


"""
__Model__

Next, we create our model, which again corresponds to a single `Gaussian` with manual priors.
"""
model = af.Collection(gaussian=af.ex.Gaussian)

model.gaussian.centre = af.UniformPrior(lower_limit=10.0, upper_limit=90.0)
model.gaussian.normalization = af.UniformPrior(lower_limit=1.0, upper_limit=100.0)
model.gaussian.sigma = af.UniformPrior(lower_limit=1.0, upper_limit=10.0)

database_size = 250

for i in range(database_size):

    dataset_name = f"gaussian_{i}"

    """
    __Data__
    """
    gaussian = model.random_instance()

    data, noise_map = simulate_line_from_gaussian(instance=gaussian)

    """
    __Analysis__
    """
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    """
    __Search__
    """
    dynesty = af.DynestyStatic(
        path_prefix=path.join("database", "profiling"),
        number_of_cores=4,
        unique_tag=dataset_name,
    )

    dynesty.fit(model=model, analysis=analysis)


"""
__Database__

First, note how the results are not contained in the `output` folder after each search completes. Instead, they are
contained in the `queries_profiling.sqlite` file, which we can load using the `Aggregator`.
"""
if path.exists(path.join("output", "profiling.sqlite")):
    os.remove(path.join("output", "profiling.sqlite"))

agg = af.Aggregator.from_database("profiling.sqlite")

start = time.time()
agg.add_directory(directory=path.join("output", "database", "profiling"))
print(f"Time to add directory to database {time.time() - start}")
