"""
Módulo de datasets para el proyecto de entrenamiento distribuido.
"""

from .mnist import get_mnist, SmallCNN_MNIST
from .cifar import get_cifar10, get_cifar100
from .imdb import get_imdb
from .wikitext import get_wikitext103

__all__ = [
    'get_mnist',
    'get_cifar10',
    'get_cifar100',
    'get_imdb',
    'get_wikitext103',
    'SmallCNN_MNIST'
]