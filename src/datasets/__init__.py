"""
Módulo de datasets para el proyecto de entrenamiento distribuido.
"""

from .mnist import get_mnist, SmallCNN_MNIST
from .cifar import get_cifar10, get_cifar100
from .imdb import get_imdb
from .wikitext import get_wikitext103
from .distributed import (
    get_mnist_distributed,
    get_cifar10_distributed,
    get_cifar100_distributed,
    get_imdb_distributed,
    get_wikitext103_distributed
)

__all__ = [
    'get_mnist',
    'get_cifar10',
    'get_cifar100',
    'get_imdb',
    'get_wikitext103',
    'SmallCNN_MNIST',
    # Distribuidos
    'get_mnist_distributed',
    'get_cifar10_distributed',
    'get_cifar100_distributed',
    'get_imdb_distributed',
    'get_wikitext103_distributed'
]