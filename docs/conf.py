# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Sphinx configuration for the lauelab documentation."""

from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
    metadata = tomllib.load(pyproject_file)["project"]

project = metadata["name"]
author = "Advanced Photon Source"
copyright = "2026, Advanced Photon Source"
release = metadata["version"]
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_llms_txt",
]

# llms.txt and llms-full.txt for AI agents (llmstxt.org). Page links resolve
# against html_baseurl, defined with the other html_* settings below.
llms_txt_title = "lauelab"
llms_txt_summary = (
    "Python bindings to native Laue diffraction indexing at APS beamline "
    "34-ID-E, with reflection simulation, orientation analysis, "
    "visualization, and wire-scan reconstruction."
)

# Render $...$ and $$...$$ in the Markdown pages as math.
myst_enable_extensions = ["dollarmath"]

source_suffix = {
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True
nitpick_ignore = [
    ("py:class", "Path"),
    ("py:class", "np.ndarray"),
    ("py:class", "lauelab.indexing._liblaue.DetectorGeometry"),
]

autodoc_member_order = "bysource"
autodoc_type_aliases = {
    "Path": "pathlib.Path",
    "np.ndarray": "numpy.ndarray",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autosummary_generate = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
intersphinx_disabled_reftypes = ["std:doc"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "lauelab documentation"
html_logo = "_static/lauelab-logo.svg"
html_favicon = "_static/lauelab-logo.svg"
html_baseurl = "https://advancedphotonsource.github.io/lauelab/"
html_theme_options = {
    "github_url": "https://github.com/AdvancedPhotonSource/lauelab",
    "navbar_align": "left",
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "AdvancedPhotonSource",
    "github_repo": "lauelab",
    "github_version": "main",
    "doc_path": "docs",
}

# These standalone pages have no section tree to show in the primary sidebar.
html_sidebars = {
    "installation": [],
    "quickstart": [],
}
