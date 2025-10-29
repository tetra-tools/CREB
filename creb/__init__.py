# creb/__init__.py
"""
CREB: Consistent Reference External Batch Harmonization for Neuroimaging Data
A Python package for harmonizing unseen neuroimaging data towards reference using
empirical Bayes methods.
"""

__version__ = "0.1.0"

from .crebLearn import crebLearn
from .crebApply import crebApply
from .creb_core import getBundleInfo, loadBundle

__all__ = [
    "crebLearn",
    "crebApply",
    "loadBundle", 
    "getBundleInfo"
]

