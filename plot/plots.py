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
a `NestPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
from os import path
import os

cwd = os.getcwd()
from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "plot"))

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
# """
search = af.DynestyStatic(path_prefix="plot", name="NestPlotter")

result = search.fit(model=model, analysis=analysis)

samples = result.samples

plotter = aplt.NestPlotter(
    samples=samples,
    output=aplt.Output(path=path.join("plot", "dynesty"), format="png"),
)

plotter.output.filename = "corner"
plotter.corner_anesthetic()
plotter.corner_cornerpy()

from dynesty import plotting as dyplot

model = result.model
search_internal = result.search_internal

dyplot.boundplot(
    results=search_internal.results,
    labels=model.parameter_labels_with_superscripts_latex,
    dims=(2, 2),
    it=100,
)

plt.savefig(path.join("plot", "dynesty", "boundplot.png"))
plt.close()

dyplot.cornerbound(
    results=search_internal.results,
    labels=model.parameter_labels_with_superscripts_latex,
    it=100,
)

plt.savefig(path.join("plot", "dynesty", "cornerbound.png"))
plt.close()

dyplot.cornerplot(
    results=search_internal.results,
    labels=model.parameter_labels_with_superscripts_latex,
)

plt.savefig(path.join("plot", "dynesty", "cornerplot.png"))
plt.close()

dyplot.cornerpoints(
    results=search_internal.results,
    labels=model.parameter_labels_with_superscripts_latex,
)

plt.savefig(path.join("plot", "dynesty", "cornerpoints.png"))
plt.close()

dyplot.runplot(
    results=search_internal.results,
)

plt.savefig(path.join("plot", "dynesty", "runplot.png"))
plt.close()

dyplot.traceplot(
    results=search_internal.results,
)

plt.savefig(path.join("plot", "dynesty", "traceplot.png"))
plt.close()

"""
__Nautilus__
"""
# search = af.Nautilus(
#     path_prefix="plot",
#     name="NestPlotter",
#     n_live=100,  # Number of so-called live points. New bounds are constructed so that they encompass the live points.
# )
#
# result = search.fit(model=model, analysis=analysis)
#
# samples = result.samples
#
# plotter = aplt.NestPlotter(
#     samples=samples,
#     output=aplt.Output(path=path.join("plot", "nautilus"), format="png"),
# )
#
# plotter.output.filename = "corner"
# plotter.corner_anesthetic()


"""
__UltraNest__
"""
# search = af.UltraNest(path_prefix="plot", name="NestPlotter")
#
# result = search.fit(model=model, analysis=analysis)
#
# samples = result.samples
#
# plotter = aplt.NestPlotter(
#     samples=samples,
#     output=aplt.Output(path=path.join("plot", "ultranest"), format="png"),
# )
#
# plotter.output.filename = "corner"
# plotter.corner_anesthetic()


"""
__Emcee__
"""
search = af.Emcee(
    path_prefix=path.join("plot"), name="MCMCPlotter", nwalkers=100, nsteps=500
)

result = search.fit(model=model, analysis=analysis)

samples = result.samples

plotter = aplt.MCMCPlotter(
    samples=samples,
    output=aplt.Output(path=path.join("plot", "emcee"), format="png"),
)

plotter.output.filename = "corner"
plotter.corner_cornerpy()

search_internal = result.search_internal

"""
The method below shows a 2D projection of the walker trajectories.
"""
fig, axes = plt.subplots(result.model.prior_count, figsize=(10, 7))

for i in range(result.model.prior_count):
    for walker_index in range(search_internal.get_log_prob().shape[1]):
        ax = axes[i]
        ax.plot(
            search_internal.get_chain()[:, walker_index, i],
            search_internal.get_log_prob()[:, walker_index],
            alpha=0.3,
        )

    ax.set_ylabel("Log Likelihood")
    ax.set_xlabel(result.model.parameter_labels_with_superscripts_latex[i])

plt.savefig(path.join("plot", "emcee", "tracjectories.png"))

"""
This method shows the likelihood as a series of steps.
"""

fig, axes = plt.subplots(1, figsize=(10, 7))

for walker_index in range(search_internal.get_log_prob().shape[1]):
    axes.plot(search_internal.get_log_prob()[:, walker_index], alpha=0.3)

axes.set_ylabel("Log Likelihood")
axes.set_xlabel("step number")

plt.savefig(path.join("plot", "emcee", "likelihood_series.png"))

"""
This method shows the parameter values of every walker at every step.
"""
fig, axes = plt.subplots(result.samples.model.prior_count, figsize=(10, 7), sharex=True)

for i in range(result.samples.model.prior_count):
    ax = axes[i]
    ax.plot(search_internal.get_chain()[:, :, i], alpha=0.3)
    ax.set_ylabel(result.model.parameter_labels_with_superscripts_latex[i])

axes[-1].set_xlabel("step number")

plt.savefig(path.join("plot", "emcee", "time_series.png"))

"""
__Zeus__
"""
search = af.Zeus(
    path_prefix=path.join("plot"), name="MCMCPlotter", nwalkers=100, nsteps=500
)

result = search.fit(model=model, analysis=analysis)

samples = result.samples

plotter = aplt.MCMCPlotter(
    samples=samples,
    output=aplt.Output(path=path.join("plot", "zeus"), format="png"),
)

plotter.output.filename = "corner"
plotter.corner_cornerpy()

plotter.output.filename = "corner"
plotter.corner_cornerpy()

search_internal = result.search_internal

"""
The method below shows a 2D projection of the walker trajectories.
"""
fig, axes = plt.subplots(result.model.prior_count, figsize=(10, 7))

for i in range(result.model.prior_count):
    for walker_index in range(search_internal.get_log_prob().shape[1]):
        ax = axes[i]
        ax.plot(
            search_internal.get_chain()[:, walker_index, i],
            search_internal.get_log_prob()[:, walker_index],
            alpha=0.3,
        )

    ax.set_ylabel("Log Likelihood")
    ax.set_xlabel(result.model.parameter_labels_with_superscripts_latex[i])

plt.savefig(path.join("plot", "zeus", "tracjectories.png"))

"""
This method shows the likelihood as a series of steps.
"""

fig, axes = plt.subplots(1, figsize=(10, 7))

for walker_index in range(search_internal.get_log_prob().shape[1]):
    axes.plot(search_internal.get_log_prob()[:, walker_index], alpha=0.3)

axes.set_ylabel("Log Likelihood")
axes.set_xlabel("step number")

plt.savefig(path.join("plot", "zeus", "likelihood_series.png"))

"""
This method shows the parameter values of every walker at every step.
"""
fig, axes = plt.subplots(result.samples.model.prior_count, figsize=(10, 7), sharex=True)

for i in range(result.samples.model.prior_count):
    ax = axes[i]
    ax.plot(search_internal.get_chain()[:, :, i], alpha=0.3)
    ax.set_ylabel(result.model.parameter_labels_with_superscripts_latex[i])

axes[-1].set_xlabel("step number")

plt.savefig(path.join("plot", "zeus", "time_series.png"))

"""
__PySwarms__
"""
search = af.PySwarmsGlobal(
    path_prefix=path.join("plot"), name="MLEPlotter", n_particles=100, iters=10
)

result = search.fit(model=model, analysis=analysis)

from pyswarms.utils import plotters

plotters.plot_contour(
    pos_history=result.search_internal.pos_history,
    title="Trajectories",
)
plt.savefig(path.join("plot", "pyswarms", "contour.png"))

plotters.plot_cost_history(
    cost_history=result.search_internal.cost_history,
    ax=None,
    title="Cost History",
    designer=None,
)
plt.savefig(path.join("plot", "pyswarms", "cost_history.png"))

"""
Finish.
"""
