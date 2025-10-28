# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dwarf-p-ice3'
copyright = '2025, Loïc Maurin'
author = 'Loïc Maurin'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",           # auto doc for docstring
    "sphinx.ext.napoleon",          # support for numpy + google style docstrings
    "sphinx.ext.mathjax",           # support for mathjax
    "sphinx.ext.autosectionlabel",  # support for section referencing
    "myst_parser",                  # support for myst markdown docs
]

templates_path = ['_templates']
exclude_patterns = []

language = 'fr'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
