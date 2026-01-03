"""
Módulo de modelos para el proyecto de entrenamiento distribuido.
"""

from .mlp import MLP
from .cnn import SmallCNN, MediumCNN
from .resnet import LargeCNN
from .transformer import SmallTransformer, MediumTransformer

__all__ = [
    'MLP',
    'SmallCNN',
    'MediumCNN',
    'LargeCNN',
    'SmallTransformer',
    'MediumTransformer'
]