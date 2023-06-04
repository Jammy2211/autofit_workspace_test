"""
Database: Introduction
======================

The default behaviour of **PyAutoCTI** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check. For small model-fitting tasks this is sufficient, however many users
have a need to perform many model fits to large samples of CTI data, making manual inspection of results time consuming.
#
PyAutoCTI's database feature outputs all model-fitting results as a
sqlite3 (https://docs.python.org/3/library/sqlite3.html) relational database, such that all results
can be efficiently loaded into a Jupyter notebook or Python script for inspection, analysis and interpretation. This
database supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset) can be
loaded.

This script fits a sample of three simulated CTI datasets using the same non-linear search. The results will be used
to illustrate the database in the database tutorials that follow.

__Model__

In this script, we fit a 1D CTI Dataset to calibrate a CTI model, where:

 - The CTI model consists of multiple parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
from os import path
import os
import autofit as af

"""
__Dataset__

For each dataset we load it from hard-disc, set up its `Analysis` class and fit it with a non-linear search. 

We want each results to be stored in the database with an entry specific to the dataset. We'll fit 3 different
`simple` datasets separately 3 times in order to do this.
"""
dataset_name_list = ["simple", "simple", "simple"]

# norm_list = [100, 5000, 25000, 200000]
# norm_list = [100] # No Error
norm_list = [100, 5000]  # Flush Error

"""
__Dataset Quantities__

The standard quantities we use to load datasets before model-fitting.
"""
pixel_scales = 0.1
shape_native = (200,)
prescan = ac.Region1D(region=(0, 10))
overscan = ac.Region1D(region=(190, 200))
region_list = [(10, 20)]

total_datasets = len(norm_list)

"""
___Session__

To output results directly to the database, we start a session, which includes the name of the database `.sqlite` file
where results are stored.

# NOTE: Writing directly to database via a session is currently not supported, therefore we build results
via the scrape function below.
"""
# session = af.db.open_database("database.sqlite")
session = None

for i, dataset_name in enumerate(dataset_name_list):
    """
    __Paths__

    Set up the config and output paths.
    """
    dataset_path = path.join("dataset", "dataset_1d", dataset_name)

    """
    __Layout__

    We use the regions above to create the `Layout1D` of every 1D CTI dataset we fit. This is used  for visualizing 
    the model-fit.
    """
    layout_list = [
        ac.Layout1D(
            shape_1d=shape_native,
            region_list=region_list,
            prescan=prescan,
            overscan=overscan,
        )
        for i in range(total_datasets)
    ]

    """
    __Dataset__

    We now load every cti-dataset, including a noise-map and pre-cti data containing the data before read-out and
    therefore without CTI.
    """
    dataset_list = [
        ac.Dataset1D.from_fits(
            data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
            noise_map_path=path.join(
                dataset_path, f"norm_{int(norm)}", "noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
            ),
            layout=layout,
            pixel_scales=0.1,
        )
        for layout, norm in zip(layout_list, norm_list)
    ]

    """
    __Mask__

    We now mask every 1D dataset, removing the FPR of each dataset so we use only the EPER to calibrate the CTI model.
    """
    mask = ac.Mask1D.all_false(
        shape_slim=dataset_list[0].shape_slim,
        pixel_scales=dataset_list[0].pixel_scales,
    )

    mask = ac.Mask1D.masked_fpr_and_eper_from(
        mask=mask,
        layout=dataset_list[0].layout,
        settings=ac.SettingsMask1D(fpr_pixels=(0, 10)),
        pixel_scales=dataset_list[0].pixel_scales,
    )

    dataset_list = [dataset.apply_mask(mask=mask) for dataset in dataset_list]

    """
    __Info__

    Information about our model-fit that isn't part of the model-fit can be made accessible to the database, by 
    passing an `info` dictionary. 

    Below we load this info dictionary from an `info.json` file stored in each dataset's folder. This dictionary
    contains the (hypothetical) injection voltage settings of the data, as if it were made via a charge injection.
    """
    info_list = []

    for norm in norm_list:
        info_file = path.join(dataset_path, f"norm_{int(norm)}", "info.json")

        with open(info_file) as json_file:
            info = json.load(json_file)

        info_list.append(info)

    """
    __Model__

    Set up the model as per usual.
    """
    trap_0 = af.Model(ac.TrapInstantCapture)
    trap_1 = af.Model(ac.TrapInstantCapture)

    # Bug means cant combine session with assertion for now.

    # trap_0.add_assertion(trap_0.release_timescale < trap_1.release_timescale)

    trap_list = [trap_0, trap_1]

    ccd = af.Model(ac.CCDPhase)
    ccd.well_notch_depth = 0.0
    ccd.full_well_depth = 200000.0

    model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

    """
    __Clocker / arCTIc__

    Set up the clocker as per usual.
    """
    clocker_1d = ac.Clocker1D(express=5)

    """
    In all examples so far, results were written to the `autofit_workspace/output` folder with a path and folder 
    named after a unique identifier, which was derived from the non-linear search and model. This unique identifier
    plays a vital role in the database: it is used to ensure every entry in the database is unique. 

    In this example, results are written directly to the `database.sqlite` file after the model-fit is complete and 
    only stored in the output folder during the model-fit. This can be important for performing large model-fitting 
    tasks on high performance computing facilities where there may be limits on the number of files allowed, or there
    are too many results to make navigating the output folder manually feasible.

    The `unique_tag` below uses the `dataset_name` to alter the unique identifier, which as we have seen is also 
    generated depending on the search settings and model. In this example, all three model fits use an identical 
    search and model, so this `unique_tag` is key for ensuring 3 separate sets of results for each model-fit are 
    stored in the output folder and written to the .sqlite database. 
    """
    search = af.DynestyStatic(
        path_prefix=path.join("database"),
        name="database_example",
        unique_tag=f"{dataset_name}_{i}",  # This makes the unique identifier use the dataset name
        session=session,  # This instructs the search to write to the .sqlite database.
        nlive=50,
    )

    analysis_list = [
        ac.AnalysisDataset1D(dataset=dataset, clocker=clocker_1d)
        for dataset in dataset_list
    ]
    analysis = sum(analysis_list)

    search.fit(analysis=analysis, model=model,
             #  info=info
               )

"""
If you inspect the `autocti_workspace/output/database` folder during the model-fit, you'll see that the results
are only stored there during the model fit, and they are written to the database and removed once complete. 

__Loading Results__

Note: This would normally be presented as optional, but the database currently requires us to use this method to
load results correctly.

After fitting a large suite of data, we can use the aggregator to load the database's results. We can then
manipulate, interpret and visualize them using a Python script or Jupyter notebook.

The results are not contained in the `output` folder after each search completes. Instead, they are
contained in the `database.sqlite` file, which we can load using the `Aggregator`.
"""
database_file = "database.sqlite"
# agg = af.Aggregator.from_database(filename=database_file)

database_name = "database"

try:
    os.remove(path.join("output", database_file))
except FileNotFoundError:
    pass

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False, top_level_only=False
)

agg.add_directory(directory=path.join("output", database_name))

"""
__Generators__

Before using the aggregator to inspect results, let me quickly cover Python generators. A generator is an object that 
iterates over a function when it is called. The aggregator creates all of the objects that it loads from the database 
as generators (as opposed to a list, or dictionary, or other Python type).

Why? Because lists and dictionaries store every entry in memory simultaneously. If you fit many datasets, this will use 
a lot of memory and crash your laptop! On the other hand, a generator only stores the object in memory when it is used; 
Python is then free to overwrite it afterwards. Thus, your laptop won't crash!

There are two things to bare in mind with generators:

 1) A generator has no length and to determine how many entries it contains you first must turn it into a list.

 2) Once we use a generator, we cannot use it again and need to remake it. For this reason, we typically avoid 
 storing the generator as a variable and instead use the aggregator to create them on use.

We can now create a `samples` generator of every fit. The `results` example scripts show how  
the `Samples` class acts as an interface to the results of the non-linear search.
"""
samples_gen = agg.values("samples")

"""
When we print this the length of this generator converted to a list of outputs we see 3 different `SamplesDynesty`
instances. 

These correspond to each fit of each search to each of our 3 images.
"""
print("NestedSampler Samples: \n")
print(samples_gen)
print()
print("Total Samples Objects = ", len(agg), "\n")

"""
Therefore, by loading the `Samples` via the database we can now access the results of the fit to each dataset.

For example, we can plot the maximum likelihood model for each of the 3 model-fits performed.
"""
ml_vector = [
    samps.max_log_likelihood(as_instance=False) for samps in agg.values("samples")
]

print("Max Log Likelihood Model Parameter Lists: \n")
print(ml_vector, "\n\n")

"""
__Building a Database File From an Output Folder__

The fits above directly wrote the results to the .sqlite file, which we loaded above. However, you may have results
already written to hard-disk in an output folder, which you wish to build your .sqlite file from.

This can be done via the following code, which is commented out below to avoid us deleting the existing .sqlite file.

Below, the `database_name` corresponds to the name of your output folder and is also the name of the `.sqlite` file
that is created.

If you are fitting a relatively small number of datasets (e.g. 10-100) having all results written
to hard-disk (e.g. for quick visual inspection) but using the database for sample-wide analysis may be benefitial.
"""
database_name = "database"

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False, top_level_only=False
)

agg.add_directory(directory=path.join("output", database_name))

"""
__Wrap Up__

This example illustrates how to use the database.

The `database/examples` folder contains examples illustrating the following:

- ``samples.py``: Loads the non-linear search results from the SQLite3 database and inspect the 
   samples (e.g. parameter estimates, posterior).

- ``queries.py``: Query the database to get certain modeling results (e.g. all models where `
   einstein_radius > 1.0`).

- ``models.py``: Inspect the models in the database (e.g. visualize their deflection angles).

- ``data_fitting.py``: Inspect the data-fitting results in the database (e.g. visualize the residuals).
"""
