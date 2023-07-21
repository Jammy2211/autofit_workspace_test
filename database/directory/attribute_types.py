"""
Feature: Attribute Types
========================

The `save_attributes` function outputs files to a `files` folder, which can be loaded via the database for
inspection.

This test script uses an `Analysis` class which outputs all supported data format types to the `files` folder,
and checks they can be loaded via the database.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af

import json
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

"""
__Analayis__

The Analysis below manually write out data for all filetypes supported for loading.
"""
from autoconf.dictable import Dictable

class ExampleJSonDict(Dictable):

    def __init__(self, value : float = 1.0):

        self.value = value

class Analysis(af.ex.Analysis):

    def save_attributes(self, paths: af.DirectoryPaths):

        ### JSON (Not Dictable) ###

        with open(paths._files_path / "json_example.json", "w+") as f:
            json.dump(self.data.tolist(), f, indent=4)

        ### JSON (Dictable) ###

        json_dictable = ExampleJSonDict(value=1.0)
        json_dictable.output_to_json(file_path=paths._files_path / "json_dictable_example.json")

        ### PICKLE ###

        paths.save_object(name="pickle_example.pickle", obj=self.data)

        ### CSV ###

        ### TODO : Rich Add ###

        name = "csv_example"
        csv_arr = 2.0 * np.ones(shape=(2,2))

        ### FITS ###

        from astropy.io import fits

        new_hdr = fits.Header()
        hdu = fits.PrimaryHDU(3.0 * np.ones(shape=(2,2)), new_hdr)
        hdu.writeto(paths._files_path / "fits_example.fits", overwrite=True)

analysis = Analysis(data=data, noise_map=noise_map)

"""
__Search + Fit__
"""
search = af.DynestyStatic(
    name="attribute_types",
    path_prefix=path.join("database", "directory"),
    number_of_cores=1,
    unique_tag=dataset_name,
    session=session,
)

result = search.fit(model=model, analysis=analysis)

"""
__Database (build va scrape)__

"""
from autofit.database.aggregator import Aggregator

database_file = "database_directory_general.sqlite"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass


agg = Aggregator.from_database(path.join(database_file))
agg.add_directory(
    directory=path.join("output", "database", "directory", dataset_name, "attribute_types")
)

print(agg.values("json_example"))

assert agg.values("json_dictable_example") is ExampleJSonDict
assert (agg.values("pickle_example") == data).all()
assert (agg.values("csv_example") == 2.0 * np.ones(shape=(2,2))).all()
assert (agg.values("fits_example") == 3.0 * np.ones(shape=(2,2))).all()