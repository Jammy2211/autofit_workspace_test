import autofit as af
import profiles

from os import path
import numpy as np
import matplotlib.pyplot as plt


def simulate_line_from_gaussian(gaussian, dataset_path):

    """
    Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated using and
    thus defining the number of data-points in our data.
    """
    pixels = 100
    xvalues = np.arange(pixels)

    """Evaluate this `Gaussian` model instance at every xvalues to create its model profile."""
    model_line = gaussian.profile_from_xvalues(xvalues=xvalues)

    """Determine the noise (at a specified signal to noise level) in every pixel of our model profile."""
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

    """
    Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio to compute
    noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
    """
    data = model_line + noise
    noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

    """
    Output the data and noise-map to the `autofit_workspace/dataset` folder so they can be loaded and used 
    in other example scripts.
    """
    af.util.numpy_array_to_json(
        array=data, file_path=path.join(dataset_path, "data.json"), overwrite=True
    )
    af.util.numpy_array_to_json(
        array=noise_map,
        file_path=path.join(dataset_path, "noise_map.json"),
        overwrite=True,
    )
    plt.errorbar(
        x=xvalues,
        y=data,
        yerr=noise_map,
        color="k",
        ecolor="k",
        elinewidth=1,
        capsize=2,
    )
    plt.title("1D Gaussian Dataset.")
    plt.xlabel("x values of profile")
    plt.ylabel("Profile normalization")
    plt.savefig(path.join(dataset_path, "image.png"))
    plt.close()
