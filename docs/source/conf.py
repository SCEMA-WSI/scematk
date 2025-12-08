# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Add project root to sys.path so autodoc can import the package without installing it
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'scematk'
copyright = '2025, Hugh Warden'
author = 'Hugh Warden'
release = '0.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # optional: Google/NumPy style docstrings
]

autosummary_generate = True  # IMPORTANT: makes autosummary build pages automatically

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    'inherited-members': True,
    "show-inheritance": True,
}

# Avoid importing heavy/optional dependencies on Read the Docs
autodoc_mock_imports = [
    "csbdeep",
    "dask",
    "dask_image",
    "dask_jobqueue",
    "matplotlib",
    "matplotlib_scalebar",
    "numpy",
    "openslide",
    "requests",
    "skimage",
    "scipy",
    "shapely",
    "stardist",
    "tensorflow",
    "tqdm",
    "zarr",
]
