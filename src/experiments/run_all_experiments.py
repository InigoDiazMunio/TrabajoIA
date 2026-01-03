"""
Script principal para ejecutar todos los experimentos sin paralelización.
"""

import torch
import sys
sys.path.insert(0, '.')  # Ajustar según donde ejecutes

from src.datasets import get_mnist, get_cifar10, get_cifar100, get_imdb, get_wikitext103, SmallCNN_MNIST
from src.models import MLP, SmallCNN, MediumCNN, LargeCNN, SmallTransformer, MediumTransformer
from src.training import train_model


def main():
    """Función principal que ejecuta todos los experimentos"""
    
    # Detectar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Usando dispositivo: {device}")
    print(f"{'='*70}\n")
    
    # Diccionario para guardar resultados
    resultados = {}
    
    # ========== EXPERIMENTO 1: MNIST + MLP ==========
    print("\n" + "="*70)
    print("EXPERIMENTO 1: MNIST + MLP")
    print("="*70)
    train_loader, test_loader = get_mnist(batch_size=64)
    model = MLP()
    resultados['MNIST + MLP'] = train_model(
        model, 
        train_loader, 
        test_loader, 
        epochs=5, 
        device=device, 
        name="MNIST + MLP"
    )
    
    # ========== EXPERIMENTO 2: MNIST + CNN Small ==========
    print("\n" + "="*70)
    print("EXPERIMENTO 2: MNIST + CNN Small")
    print("="*70)
    train_loader, test_loader = get_mnist(batch_size=64)
    model = SmallCNN_MNIST()
    resultados['MNIST + CNN Small'] = train_model(
        model,
        train_loader,
        test_loader,
        epochs=5,
        device=device,
        name="MNIST + CNN Small"
    )
    
    # ========== EXPERIMENTO 3: CIFAR-10 + CNN Medium ==========
    print("\n" + "="*70)
    print("EXPERIMENTO 3: CIFAR-10 + CNN Medium")
    print("="*70)
    train_loader, test_loader = get_cifar10(batch_size=64)
    model = MediumCNN()
    resultados['CIFAR-10 + CNN Medium'] = train_model(
        model,
        train_loader,
        test_loader,
        epochs=5,
        device=device,
        name="CIFAR-10 + CNN Medium"
    )
    
    # ========== EXPERIMENTO 4: CIFAR-100 + CNN Large ==========
    print("\n" + "="*70)
    print("EXPERIMENTO 4: CIFAR-100 + CNN Large")
    print("="*70)
    train_loader, test_loader = get_cifar100(batch_size=64)
    model = LargeCNN(num_classes=100)
    resultados['CIFAR-100 + CNN Large'] = train_model(
        model,
        train_loader,
        test_loader,
        epochs=10,
        device=device,
        name="CIFAR-100 + CNN Large"
    )
    
    # ========== EXPERIMENTO 5: IMDB + Transformer Small ==========
    print("\n" + "="*70)
    print("EXPERIMENTO 5: IMDB + Transformer Small")
    print("="*70)
    train_loader, test_loader, vocab_size = get_imdb(batch_size=32)
    model = SmallTransformer(vocab_size=vocab_size)
    resultados['IMDB + Transformer Small'] = train_model(
        model,
        train_loader,
        test_loader,
        epochs=3,
        device=device,
        name="IMDB + Transformer Small",
        is_transformer=True
    )
    
    # ========== EXPERIMENTO 6: WikiText-103 + Transformer Medium ==========
    print("\n" + "="*70)
    print("EXPERIMENTO 6: WikiText-103 + Transformer Medium")
    print("="*70)
    train_loader, test_loader, vocab_size = get_wikitext103(batch_size=16, subset_size=5000)
    model = MediumTransformer(vocab_size=vocab_size)
    resultados['WikiText + Transformer Medium'] = train_model(
        model,
        train_loader,
        test_loader,
        epochs=3,
        device=device,
        name="WikiText + Transformer Medium",
        is_transformer=True,
        is_lm=True
    )
    
    # ========== RESUMEN FINAL ==========
    print("\n" + "="*70)
    print("RESUMEN DE TODOS LOS EXPERIMENTOS")
    print("="*70)
    for nombre, metricas in resultados.items():
        print(f"\n{nombre}:")
        print(f"  Tiempo: {metricas['tiempo']:.2f}s")
        print(f"  Throughput: {metricas['throughput']:.2f} samples/s")
        if 'WikiText' in nombre:
            print(f"  Mejor Perplexity: {metricas['accuracy']:.2f}")
        else:
            print(f"  Mejor Accuracy: {metricas['accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("✅ TODOS LOS EXPERIMENTOS COMPLETADOS")
    print("="*70 + "\n")
    
    return resultados


if __name__ == "__main__":
    resultados = main()