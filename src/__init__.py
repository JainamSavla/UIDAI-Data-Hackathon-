"""
UIDAI Hackathon Analytics Pipeline - Source Package
Complete modular analytics framework for Aadhaar enrollment analysis
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"
__description__ = "UIDAI Hackathon Analytics Pipeline"

from . import utils
from . import data_loader
from . import preprocessing
from . import feature_engineering
from . import descriptive_analysis
from . import diagnostic_analysis
from . import predictive_models
from . import prescriptive_optimization
from . import visualization

__all__ = [
    'utils',
    'data_loader',
    'preprocessing',
    'feature_engineering',
    'descriptive_analysis',
    'diagnostic_analysis',
    'predictive_models',
    'prescriptive_optimization',
    'visualization'
]
