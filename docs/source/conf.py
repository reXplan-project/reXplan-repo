# -- Path setup --------------------------------------------------------------

import os, sys
sys.path.insert(0, os.path.abspath('../reXplan_fork'))

# -- Project information -----------------------------------------------------

project = 'reXplan'
copyright = '2023, Tim'
author = 'Tim'

release = '0.1'

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
]

templates_path = ['_templates']

exclude_patterns = []

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']