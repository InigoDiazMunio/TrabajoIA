"""
Script para ejecutar experimentos con DistributedDataParallel (DDP)

NOTA: DDP requiere al menos 2 GPUs para funcionar correctamente.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.multiprocessing as mp
from src.datasets import (
    get_mnist_distributed,
    get_cifar10_distributed,
    get_cifar100_distributed,
    get_imdb_distributed,
    get_wikitext103_distributed,
    SmallCNN_MNIST
)
from src.models import MLP, MediumCNN, LargeCNN, SmallTransformer, MediumTransformer
from src.training import train_ddp_worker


def run_experiment_ddp(name, model_fn, dataset_fn, epochs, batch_size,
                       is_transformer=False, is_lm=False, dataset_args=None):
    """
    Ejecuta un experimento con DDP.
    
    Args:
        name: Nombre del experimento
        model_fn: Función que retorna el modelo
        dataset_fn: Función de dataset distribuido
        epochs: Número de épocas
        batch_size: Tamaño del batch
        is_transformer: Si es transformer
        is_lm: Si es language modeling
        dataset_args: Argumentos extra para el dataset
    """
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print(f"\n⚠️  ADVERTENCIA: Solo {world_size} GPU disponible.")
        print("   DDP requiere al menos 2 GPUs para ser efectivo.")
        print("   Los resultados no mostrarán speedup real.\n")
    
    # Crear funciones de loader para cada rank
    def train_loader_fn(rank, ws):
        if dataset_args:
            loaders = dataset_fn(batch_size, rank, ws, *dataset_args)
        else:
            loaders = dataset_fn(batch_size, rank, ws)
        return loaders[0]  # train_loader
    
    def test_loader_fn(rank, ws):
        if dataset_args:
            loaders = dataset_fn(batch_size, rank, ws, *dataset_args)
        else:
            loaders = dataset_fn(batch_size, rank, ws)
        return loaders[1]  # test_loader
    
    # Lanzar procesos DDP
    mp.spawn(
        train_ddp_worker,
        args=(world_size, model_fn, train_loader_fn, test_loader_fn,
              epochs, name, is_transformer, is_lm),
        nprocs=world_size,
        join=True
    )
    
    print(f"\n✅ Experimento '{name}' completado\n")


def main():
    """Ejecuta todos los experimentos con DDP"""
    
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*70}")
    print(f"ENTRENAMIENTO DISTRIBUIDO (DDP)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Número de GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"{'='*70}\n")
    
    if num_gpus < 2:
        print("⚠️  ADVERTENCIA CRÍTICA:")
        print("   DDP está diseñado para multi-GPU (2+ GPUs)")
        print("   Con 1 GPU no habrá speedup real.")
        print("   Para multi-GPU en Colab: Runtime → Change runtime type → GPU\n")
        response = input("¿Deseas continuar de todos modos? (s/n): ")
        if response.lower() != 's':
            print("Ejecución cancelada.")
            return
    
    # EXPERIMENTO 1: MNIST + MLP
    print("\n" + "="*70)
    print("EXPERIMENTO 1: MNIST + MLP (DDP)")
    print("="*70)
    run_experiment_ddp(
        name="MNIST + MLP",
        model_fn=MLP,
        dataset_fn=get_mnist_distributed,
        epochs=5,
        batch_size=64
    )
    
    # EXPERIMENTO 2: MNIST + CNN Small
    print("\n" + "="*70)
    print("EXPERIMENTO 2: MNIST + CNN Small (DDP)")
    print("="*70)
    run_experiment_ddp(
        name="MNIST + CNN Small",
        model_fn=SmallCNN_MNIST,
        dataset_fn=get_mnist_distributed,
        epochs=5,
        batch_size=64
    )
    
    # EXPERIMENTO 3: CIFAR-10 + CNN Medium
    print("\n" + "="*70)
    print("EXPERIMENTO 3: CIFAR-10 + CNN Medium (DDP)")
    print("="*70)
    run_experiment_ddp(
        name="CIFAR-10 + CNN Medium",
        model_fn=lambda: MediumCNN(num_classes=10),
        dataset_fn=get_cifar10_distributed,
        epochs=5,
        batch_size=64
    )
    
    # EXPERIMENTO 4: CIFAR-100 + CNN Large
    print("\n" + "="*70)
    print("EXPERIMENTO 4: CIFAR-100 + CNN Large (DDP)")
    print("="*70)
    run_experiment_ddp(
        name="CIFAR-100 + CNN Large",
        model_fn=lambda: LargeCNN(num_classes=100),
        dataset_fn=get_cifar100_distributed,
        epochs=10,
        batch_size=64
    )
    
    # EXPERIMENTO 5: IMDB + Transformer Small
    print("\n" + "="*70)
    print("EXPERIMENTO 5: IMDB + Transformer Small (DDP)")
    print("="*70)
    
    # Para IMDB necesitamos crear el modelo con vocab_size
    from datasets import load_dataset
    from transformers import AutoTokenizer
    dataset = load_dataset('imdb')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    run_experiment_ddp(
        name="IMDB + Transformer Small",
        model_fn=lambda: SmallTransformer(vocab_size=vocab_size, num_classes=2),
        dataset_fn=get_imdb_distributed,
        epochs=3,
        batch_size=32,
        is_transformer=True
    )
    
    # EXPERIMENTO 6: WikiText-103 + Transformer Medium
    print("\n" + "="*70)
    print("EXPERIMENTO 6: WikiText-103 + Transformer Medium (DDP)")
    print("="*70)
    
    # Para WikiText obtenemos vocab_size
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    vocab_size = tokenizer.vocab_size
    
    run_experiment_ddp(
        name="WikiText + Transformer Medium",
        model_fn=lambda: MediumTransformer(vocab_size=vocab_size),
        dataset_fn=get_wikitext103_distributed,
        epochs=3,
        batch_size=16,
        is_transformer=True,
        is_lm=True,
        dataset_args=(256, 5000)  # max_length, subset_size
    )
    
    print("\n" + "="*70)
    print("✅ TODOS LOS EXPERIMENTOS DDP COMPLETADOS")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Necesario para Windows
    mp.set_start_method('spawn', force=True)
    main()