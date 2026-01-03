"""
Módulo de entrenamiento para diferentes configuraciones de paralelización.
"""

from .single_gpu import train_model

__all__ = [
    'train_model'
]