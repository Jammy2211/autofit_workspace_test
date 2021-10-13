"""
__Imports__
"""
import os
from os import path
import warnings

warnings.filterwarnings("ignore")

from autoconf import conf

"""
__Paths__
"""
workspace_path = os.getcwd()

config_path = path.join(workspace_path, "config")
conf.instance.push(new_path=config_path)

"""
___Database__

The name of the database, which corresponds to the output results folder.
"""
output_directory = path.join("id_update", "output", "updated")
map_filename = path.join("id_update", "update.yaml")

from autofit.tools.update_identifiers import update_identifiers_from_file

update_identifiers_from_file(
    output_directory=output_directory, map_filename=map_filename
)
