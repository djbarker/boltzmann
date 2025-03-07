# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Boltzmann"
copyright = "2025, Daniel Barker"
author = "Daniel Barker"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    # "sphinxcontrib.apidoc",
]

autodoc_member_order = "bysource"

# see: https://github.com/sphinx-contrib/apidoc
# apidoc_force = True
# # apidoc_module_first = True
# apidoc_module_dir = "../../python/boltzmann"
# apidoc_output_dir = "api"
# apidoc_excluded_paths = []  # e.g. tests
# apidoc_separate_modules = True


templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinxdoc"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
