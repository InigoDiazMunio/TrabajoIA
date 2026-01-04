"""
Módulo de entrenamiento para diferentes configuraciones de paralelización.

Disponibles:
- single_gpu: Entrenamiento baseline sin paralelización
- data_parallel: DataParallel (nn.DataParallel)
- distributed: DistributedDataParallel (DDP)
"""

from .single_gpu import train_model
from .data_parallel import train_model_dataparallel
from .distributed import train_model_ddp, setup_ddp, cleanup, is_main_process

__all__ = [
    'train_model',
    'train_model_dataparallel',
    'train_model_ddp',
    'setup_ddp',
    'cleanup',
    'is_main_process'
]