"""
Script para ejecutar todos los experimentos con DataParallel
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.datasets import get_mnist, get_cifar10, get_cifar100, get_imdb, get_wikitext103, SmallCNN_MNIST
from src.models import MLP, SmallCNN, MediumCNN, LargeCNN, SmallTransformer, MediumTransformer
from src.training import train_model_dataparallel


def save_results(resultados, filename=None):
    """Guarda los resultados en JSON y Excel."""
    os.makedirs('results', exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"results/experimentos_dataparallel_{timestamp}"
    else:
        base_filename = f"results/{filename}"
    
    # Guardar JSON
    json_filename = base_filename + ".json"
    resultados_serializables = {}
    for nombre, metricas in resultados.items():
        resultados_serializables[nombre] = {
            'tiempo': metricas['tiempo'],
            'throughput': metricas['throughput'],
            'accuracy': metricas['accuracy'],
            'memoria_gb': metricas['memoria_gb'],
            'num_gpus': metricas['num_gpus'],
            'epoch_logs': metricas['epoch_logs']
        }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(resultados_serializables, f, indent=2, ensure_ascii=False)
    print(f"\n💾 JSON guardado en: {json_filename}")
    
    # Guardar Excel
    excel_filename = base_filename + ".xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Hoja 1: Resumen general
        resumen_data = []
        for nombre, metricas in resultados.items():
            resumen_data.append({
                'Experimento': nombre,
                'Tiempo (s)': round(metricas['tiempo'], 2),
                'Throughput (samples/s)': round(metricas['throughput'], 2),
                'Mejor Accuracy/Perplexity': round(metricas['accuracy'], 4),
                'Memoria GPU (GB)': round(metricas['memoria_gb'], 2),
                'Num GPUs': metricas['num_gpus']
            })
        
        df_resumen = pd.DataFrame(resumen_data)
        df_resumen.to_excel(writer, sheet_name='Resumen', index=False)
        
        # Hoja 2: Logs por época
        logs_data = []
        for nombre, metricas in resultados.items():
            for log in metricas['epoch_logs']:
                row = {
                    'Experimento': nombre,
                    'Época': log['epoch'],
                    'Loss': round(log['loss'], 4)
                }
                if 'accuracy' in log:
                    row['Accuracy'] = round(log['accuracy'], 4)
                    row['Perplexity'] = None
                else:
                    row['Accuracy'] = None
                    row['Perplexity'] = round(log['perplexity'], 2)
                logs_data.append(row)
        
        df_logs = pd.DataFrame(logs_data)
        df_logs.to_excel(writer, sheet_name='Logs por Época', index=False)
    
    print(f"📊 Excel guardado en: {excel_filename}")
    print(f"✅ Todos los archivos guardados en la carpeta 'results/'")


def main():
    """Función principal que ejecuta todos los experimentos con DataParallel"""
    
    # Verificar GPUs
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Número de GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"{'='*70}\n")
    
    if num_gpus < 2:
        print("⚠️  ADVERTENCIA: Solo hay 1 GPU. DataParallel no tendrá speedup significativo.")
        print("   Para multi-GPU en Colab: Runtime → Change runtime type → GPU")
    
    resultados = {}
    
    # EXPERIMENTO 1: MNIST + MLP
    print("\n" + "="*70)
    print("EXPERIMENTO 1: MNIST + MLP (DataParallel)")
    print("="*70)
    train_loader, test_loader = get_mnist(batch_size=64)
    model = MLP()
    resultados['MNIST + MLP'] = train_model_dataparallel(
        model, train_loader, test_loader,
        epochs=5, device=device, name="MNIST + MLP"
    )
    
    # EXPERIMENTO 2: MNIST + CNN Small
    print("\n" + "="*70)
    print("EXPERIMENTO 2: MNIST + CNN Small (DataParallel)")
    print("="*70)
    train_loader, test_loader = get_mnist(batch_size=64)
    model = SmallCNN_MNIST()
    resultados['MNIST + CNN Small'] = train_model_dataparallel(
        model, train_loader, test_loader,
        epochs=5, device=device, name="MNIST + CNN Small"
    )
    
    # EXPERIMENTO 3: CIFAR-10 + CNN Medium
    print("\n" + "="*70)
    print("EXPERIMENTO 3: CIFAR-10 + CNN Medium (DataParallel)")
    print("="*70)
    train_loader, test_loader = get_cifar10(batch_size=64)
    model = MediumCNN()
    resultados['CIFAR-10 + CNN Medium'] = train_model_dataparallel(
        model, train_loader, test_loader,
        epochs=5, device=device, name="CIFAR-10 + CNN Medium"
    )
    
    # EXPERIMENTO 4: CIFAR-100 + CNN Large
    print("\n" + "="*70)
    print("EXPERIMENTO 4: CIFAR-100 + CNN Large (DataParallel)")
    print("="*70)
    train_loader, test_loader = get_cifar100(batch_size=64)
    model = LargeCNN(num_classes=100)
    resultados['CIFAR-100 + CNN Large'] = train_model_dataparallel(
        model, train_loader, test_loader,
        epochs=10, device=device, name="CIFAR-100 + CNN Large"
    )
    
    # EXPERIMENTO 5: IMDB + Transformer Small
    print("\n" + "="*70)
    print("EXPERIMENTO 5: IMDB + Transformer Small (DataParallel)")
    print("="*70)
    train_loader, test_loader, vocab_size = get_imdb(batch_size=32)
    model = SmallTransformer(vocab_size=vocab_size)
    resultados['IMDB + Transformer Small'] = train_model_dataparallel(
        model, train_loader, test_loader,
        epochs=3, device=device,
        name="IMDB + Transformer Small",
        is_transformer=True
    )
    
    # EXPERIMENTO 6: WikiText-103 + Transformer Medium
    print("\n" + "="*70)
    print("EXPERIMENTO 6: WikiText-103 + Transformer Medium (DataParallel)")
    print("="*70)
    train_loader, test_loader, vocab_size = get_wikitext103(batch_size=16, subset_size=5000)
    model = MediumTransformer(vocab_size=vocab_size)
    resultados['WikiText + Transformer Medium'] = train_model_dataparallel(
        model, train_loader, test_loader,
        epochs=3, device=device,
        name="WikiText + Transformer Medium",
        is_transformer=True, is_lm=True
    )
    
    # RESUMEN FINAL
    print("\n" + "="*70)
    print("RESUMEN DE EXPERIMENTOS DATAPARALLEL")
    print("="*70)
    for nombre, metricas in resultados.items():
        print(f"\n{nombre}:")
        print(f"  Tiempo: {metricas['tiempo']:.2f}s")
        print(f"  Throughput: {metricas['throughput']:.2f} samples/s")
        print(f"  Memoria GPU: {metricas['memoria_gb']:.2f} GB")
        print(f"  GPUs usadas: {metricas['num_gpus']}")
        if 'WikiText' in nombre:
            print(f"  Mejor Perplexity: {metricas['accuracy']:.2f}")
        else:
            print(f"  Mejor Accuracy: {metricas['accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("✅ TODOS LOS EXPERIMENTOS DATAPARALLEL COMPLETADOS")
    print("="*70 + "\n")
    
    # Guardar resultados
    save_results(resultados)
    
    return resultados


if __name__ == "__main__":
    resultados = main()