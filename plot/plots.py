"""
Plotter: corner
===================

This test script iterates over all searches in autofit and outputs their corner plot visualization.

This is to ensure that all look as expected, similar and use the autofit samples object correctly.
"""
"""
Plots: DynestyPlotter
=====================

This example illustrates how to plot visualization summarizing the results of a dynesty non-linear search using
a `DynestyPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import autofit as af
import autofit.plot as aplt

"""
First, lets create a result via dynesty by repeating the simple model-fit that is performed in 
the `overview/simple/fit.py` example.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

model = af.Model(af.ex.Gaussian)

analysis = af.ex.Analysis(data=data, noise_map=noise_map)

"""
__Dynesty__
"""
search = af.DynestyStatic(path_prefix="plot", name="DynestyPlotter")

result = search.fit(model=model, analysis=analysis)

samples = result.samples

search_plotter = aplt.NestPlotter(
    samples=samples,
    output=aplt.Output(path=path.join("plot", "dynesty"), format="png"),
)

search_plotter.output.filename = "corner"
search_plotter.corner()

"""
__Nautilus__
"""
search = af.Nautilus(
    path_prefix="plot",
    name="NautilusPlotter",
    n_live=100,  # Number of so-called live points. New bounds are constructed so that they encompass the live points.
)

result = search.fit(model=model, analysis=analysis)

samples = result.samples

search_plotter = aplt.NestPlotter(
    samples=samples,
    output=aplt.Output(path=path.join("plot", "nautilus"), format="png"),
)

search_plotter.output.filename = "corner"
search_plotter.corner()


"""
__UltraNest__
"""
search = af.UltraNest(path_prefix="plot", name="UltraNestPlotter")

result = search.fit(model=model, analysis=analysis)

samples = result.samples

search_plotter = aplt.NestPlotter(
    samples=samples,
    output=aplt.Output(path=path.join("plot", "ultranest"), format="png"),
)

search_plotter.output.filename = "corner"
search_plotter.corner()



"""
__Emcee__
"""
search = af.Emcee(
    path_prefix=path.join("plot"), name="EmceePlotter", nwalkers=100, nsteps=500
)

result = search.fit(
    model=model,
    analysis=analysis)

samples = result.samples

search_plotter = aplt.MCMCPlotter(
    samples=samples,
    output=aplt.Output(path=path.join("plot", "emcee"), format="png"),
)

search_plotter.output.filename = "corner"
search_plotter.corner()

search_plotter.output.filename = "trajectories"
search_plotter.trajectories()

search_plotter.output.filename = "likelihood_series"
search_plotter.likelihood_series()

search_plotter.output.filename = "time_series"
search_plotter.time_series()


"""
__Zeus__
"""
search = af.Zeus(
    path_prefix=path.join("plot"), name="ZeusPlotter", nwalkers=100, nsteps=500
)

result = search.fit(
    model=model,
    analysis=analysis)

samples = result.samples

search_plotter = aplt.MCMCPlotter(
    samples=samples,
    output=aplt.Output(path=path.join("plot", "zeus"), format="png"),
)

search_plotter.output.filename = "corner"
search_plotter.corner()

search_plotter.output.filename = "trajectories"
search_plotter.trajectories()

search_plotter.output.filename = "likelihood_series"
search_plotter.likelihood_series()

search_plotter.output.filename = "time_series"
search_plotter.time_series()

"""
__PySwarms__
"""
search = af.PySwarmsGlobal(
    path_prefix=path.join("plot"), name="PySwarmsPlotter", n_particles=100, iters=100
)

result = search.fit(
    model=model,
    analysis=analysis)

samples = result.samples

search_plotter = aplt.OptimizePlotter(
    samples=samples,
    output=aplt.Output(path=path.join("plot", "pyswarms"), format="png"),
)

search_plotter.output.filename = "contour"
search_plotter.contour()

search_plotter.output.filename = "cost_history"
search_plotter.cost_history()

search_plotter.output.filename = "trajectories"
search_plotter.trajectories()

time_series = search_plotter.time_series()
time_series.figure()


"""
Finish.
"""
