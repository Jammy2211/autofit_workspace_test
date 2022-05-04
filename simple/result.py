"""
__Example: Result__

In this example, we'll repeat the fit of 1D data of a `Gaussian` profile with a 1D `Gaussian` model using the non-linear
search emcee and inspect the *Result* object that is returned in detail.

If you haven't already, you should checkout the files `example/model.py`,`example/analysis.py` and `example/fit.py` to
see how the fit is performed by the code below. The first section of code below is simmply repeating the commands in
`example/fit.py`, so feel free to skip over it until you his the `Result`'s section.
"""
#%matplotlib inline

import autofit as af
import autofit.plot as aplt
import model as m
import analysis as a

from os import path
import matplotlib.pyplot as plt
import numpy as np

"""
__Data__

First, lets load data of a 1D Gaussian, by loading it from a .json file in the directory 
`autofit_workspace/dataset/`, which  simulates the noisy data we fit (check it out to see how we simulate the 
data).
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
__Model__

Next, we create our model, which in this case corresponds to a single Gaussian. In model.py, you will have noted
this `Gaussian` has 3 parameters (centre, normalization and sigma). These are the free parameters of our model that the
non-linear search fits for, meaning the non-linear parameter space has dimensionality = 3.
"""
model = af.Model(m.Gaussian)

"""
Checkout `autofit_workspace/config/priors` - this config file defines the default priors of all our model
components. However, we can overwrite priors before running the `NonLinearSearch` as shown below.
"""
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.GaussianPrior(
    mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=np.inf
)

"""
__Analysis__

We now set up our Analysis, using the class described in `analysis.py`. The analysis describes how given an instance
of our model (a Gaussian) we fit the data and return a log likelihood value. For this simple example, we only have to
pass it the data and its noise-map.
"""
analysis = a.Analysis(data=data, noise_map=noise_map)

"""
Returns the non-linear object for emcee and perform the fit.
"""
emcee = af.Emcee(
    nwalkers=30,
    nsteps=1000,
    initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
    auto_correlations_settings=af.AutoCorrelationsSettings(
        check_for_convergence=True,
        check_size=100,
        required_length=50,
        change_threshold=0.01,
    ),
    number_of_cores=1,
)

result = emcee.fit(model=model, analysis=analysis)

"""
__Result__

Here, we'll look in detail at what information is contained in the result.

__Samples__

It contains a `Samples` object, which contains information on the non-linear sampling, for example the parameters. 
The parameters are stored as a list of lists, where the first entry corresponds to the sample index and second entry
the parameter index.
"""
samples = result.samples

print("Final 10 Parameters:")
print(samples.parameter_lists[-10:])

print("Sample 10`s third parameter value (Gaussian -> sigma)")
print(samples.parameter_lists[9][2], "\n")

"""
The Samples class also contains the log likelihood, log prior, log posterior and weight_list of every accepted sample, 
where:

- The log likelihood is the value evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise 
normalized).

- The log prior encodes information on how the priors on the parameters maps the log likelihood value to the log 
posterior value.

- The log posterior is log_likelihood + log_prior.

- The weight gives information on how samples should be combined to estimate the posterior. The weight values depend on 
the sampler used, for MCMC samples they are all 1 (e.g. all weighted equally).
     
Lets inspect the last 10 values of each for the analysis.     
"""
print("Final 10 Log Likelihoods:")
print(samples.log_likelihood_list[-10:])

print("Final 10 Log Priors:")
print(samples.log_prior_list[-10:])

print("Final 10 Log Posteriors:")
print(samples.log_posterior_list[-10:])

print("Final 10 Sample Weights:")
print(samples.weight_list[-10:], "\n")

"""
__Posterior__

The ``Result`` object therefore contains the full posterior information of our non-linear search, that can be used for
parameter estimation. The median pdf vector is readily available from the `Samples` object, which estimates the every
parameter via 1D marginalization of their PDFs.
"""
median_pdf_vector = samples.median_pdf_vector
print("Median PDF Vector:")
print(median_pdf_vector, "\n")

"""
The samples include methods for computing the error estimates of all parameters via 1D marginalization at an input sigma 
confidence limit. This can be returned as the size of each parameter error:
"""
error_vector_at_upper_sigma = samples.error_vector_at_upper_sigma(sigma=3.0)
error_vector_at_lower_sigma = samples.error_vector_at_lower_sigma(sigma=3.0)

print("Upper Error values (at 3.0 sigma confidence):")
print(error_vector_at_upper_sigma)

print("lower Error values (at 3.0 sigma confidence):")
print(error_vector_at_lower_sigma, "\n")

"""
They can also be returned at the values of the parameters at their error values:
"""
vector_at_upper_sigma = samples.vector_at_upper_sigma(sigma=3.0)
vector_at_lower_sigma = samples.vector_at_lower_sigma(sigma=3.0)

print("Upper Parameter values w/ error (at 3.0 sigma confidence):")
print(vector_at_upper_sigma)
print("lower Parameter values w/ errors (at 3.0 sigma confidence):")
print(vector_at_lower_sigma, "\n")

"""
__PDF__

The Probability Density Functions (PDF's) of the results can be plotted using the Emcee's visualization 
tool `corner.py`, which is wrapped via the `EmceePlotter` object.
"""
emcee_plotter = aplt.EmceePlotter(samples=result.samples)
emcee_plotter.corner()

"""
__Other Results__

The samples contain many useful vectors, including the samples with the highest likelihood and posterior values:
"""
max_log_likelihood_vector = samples.max_log_likelihood_vector
max_log_posterior_vector = samples.max_log_posterior_vector

print("Maximum Log Likelihood Vector:")
print(max_log_likelihood_vector)

print("Maximum Log Posterior Vector:")
print(max_log_posterior_vector, "\n")

"""
__Labels__

Results vectors return the results as a list, which means you need to know the parameter ordering. The list of
parameter names are available as a property of the `Samples`, as are parameter labels which can be used for labeling
figures:
"""
print(samples.model.model_component_and_parameter_names)
print(samples.model.parameter_labels)
print("\n")

"""
__Instances__

Results can instead be returned as an instance, which is an instance of the model using the Python classes used to
compose it:
"""
max_log_likelihood_instance = samples.max_log_likelihood_instance

print("Max Log Likelihood `Gaussian` Instance:")
print("Centre = ", max_log_likelihood_instance.centre)
print("Normalization = ", max_log_likelihood_instance.normalization)
print("Sigma = ", max_log_likelihood_instance.sigma, "\n")

"""
For our example problem of fitting a 1D `Gaussian` profile, this makes it straight forward to plot the maximum
likelihood model:
"""
model_data = samples.max_log_likelihood_instance.model_data_1d_via_xvalues_from(
    xvalues=np.arange(data.shape[0])
)

plt.plot(range(data.shape[0]), data)
plt.plot(range(data.shape[0]), model_data)
plt.title("Illustrative model fit to 1D `Gaussian` profile data.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

"""
All methods above are available as an instance:
"""
median_pdf_instance = samples.median_pdf_instance
instance_at_upper_sigma = samples.instance_at_upper_sigma
instance_at_lower_sigma = samples.instance_at_lower_sigma
error_instance_at_upper_sigma = samples.error_instance_at_upper_sigma
error_instance_at_lower_sigma = samples.error_instance_at_lower_sigma

"""
__Sample Instance__

A non-linear search retains every model that is accepted during the model-fit.

We can create an instance of any lens model -- below we create an instance of the 100th last accepted model.
"""
instance = samples.instance_from_sample_index(sample_index=-100)

print("Gaussian Instance of sample 100th last sample")
print("Centre = ", instance.centre)
print("Normalization = ", instance.normalization)
print("Sigma = ", instance.sigma, "\n")

"""
__Bayesian Evidence__

If a nested sampling `NonLinearSearch` is used, the evidence of the model is also available which enables Bayesian
model comparison to be performed (given we are using Emcee, which is not a nested sampling algorithm, the log evidence 
is None).:
"""
log_evidence = samples.log_evidence

"""
__Derived Errors (PDF from samples)__

Computing the errors of a quantity like the `sigma` of the Gaussian is simple, because it is sampled by the non-linear 
search. Thus, to get their errors above we used the `Samples` object to simply marginalize over all over parameters 
via the 1D Probability Density Function (PDF).

Computing the errors on a derived quantity is more tricky, because it is not sampled directly by the non-linear search. 
For example, what if we want the error on the full width half maximum (FWHM) of the Gaussian? In order to do this
we need to create the PDF of that derived quantity, which we can then marginalize over using the same function we
use to marginalize model parameters.

Below, we compute the FWHM of every accepted model sampled by the non-linear search and use this determine the PDF 
of the FWHM. When combining the FWHM's we weight each value by its `weight`. For Emcee, an MCMC algorithm, the
weight of every sample is 1, but weights may take different values for other non-linear searches.

In order to pass these samples to the function `marginalize`, which marginalizes over the PDF of the FWHM to compute 
its error, we also pass the weight list of the samples.

(Computing the error on the FWHM could be done in much simpler ways than creating its PDF from the list of every
sample. We chose this example for simplicity, in order to show this functionality, which can easily be extended to more
complicated derived quantities.)
"""
fwhm_list = []

for sample in samples.sample_list:

    instance = sample.instance_for_model(model=samples.model)

    sigma = instance.sigma

    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    fwhm_list.append(fwhm)

median_fwhm, upper_fwhm, lower_fwhm = af.marginalize(
    parameter_list=fwhm_list, sigma=3.0, weight_list=samples.weight_list
)

print(f"FWHM = {median_fwhm} ({upper_fwhm} {lower_fwhm}")

"""
__Result Extensions__

You might be wondering what else the results contains, as nearly everything we discussed above was a part of its 
`samples` property! The answer is, not much, however the result can be extended to include  model-specific results for 
your project. 

We detail how to do this in the **HowToFit** lectures, but for the example of fitting a 1D Gaussian we could extend
the result to include the maximum log likelihood profile:

(The commented out functions below are llustrative of the API we can create by extending a result).
"""
# max_log_likelihood_profile = results.max_log_likelihood_profile

"""
__Database__

For large-scaling model-fitting problems to large datasets, the results of the many model-fits performed can be output
and stored in a queryable sqlite3 database. The `Result` and `Samples` objects have been designed to streamline the 
analysis and interpretation of model-fits to large datasets using the database.

Checkout `notebooks/features/database.ipynb` for an illustration of using
this tool.
"""
