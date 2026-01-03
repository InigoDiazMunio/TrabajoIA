"""
Paquete principal del proyecto de entrenamiento distribuido de modelos de IA.
"""

__version__ = '0.1.0'
__author__ = 'Iñigo'

from . import datasets
from . import models
from . import training
from . import utils

__all__ = [
    'datasets',
    'models',
    'training',
    'utils'
]