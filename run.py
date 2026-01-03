"""
Script principal - EJECUTAR DESDE LA RAÍZ DEL PROYECTO
"""

import sys
import os
import json
from datetime import datetime
import pandas as pd

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.datasets import get_mnist, get_cifar10, get_cifar100, get_imdb, get_wikitext103, SmallCNN_MNIST
from src.models import MLP, SmallCNN, MediumCNN, LargeCNN, SmallTransformer, MediumTransformer
from src.training import train_model


def save_results(resultados, filename=None):
    """
    Guarda los resultados en JSON y Excel.
    
    Args:
        resultados: Diccionario con los resultados
        filename: Nombre base del archivo (sin extensión)
    """
    # Crear carpeta results si no existe
    os.makedirs('results', exist_ok=True)
    
    # Generar nombre de archivo con timestamp
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"results/experimentos_{timestamp}"
    else:
        base_filename = f"results/{filename}"
    
    # ========== GUARDAR JSON ==========
    json_filename = base_filename + ".json"
    
    resultados_serializables = {}
    for nombre, metricas in resultados.items():
        resultados_serializables[nombre] = {
            'tiempo': metricas['tiempo'],
            'throughput': metricas['throughput'],
            'accuracy': metricas['accuracy'],
            'epoch_logs': metricas['epoch_logs']
        }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(resultados_serializables, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 JSON guardado en: {json_filename}")
    
    # ========== GUARDAR EXCEL ==========
    excel_filename = base_filename + ".xlsx"
    
    # Crear un Excel con múltiples hojas
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        # Hoja 1: Resumen general
        resumen_data = []
        for nombre, metricas in resultados.items():
            resumen_data.append({
                'Experimento': nombre,
                'Tiempo (s)': round(metricas['tiempo'], 2),
                'Throughput (samples/s)': round(metricas['throughput'], 2),
                'Mejor Accuracy/Perplexity': round(metricas['accuracy'], 4)
            })
        
        df_resumen = pd.DataFrame(resumen_data)
        df_resumen.to_excel(writer, sheet_name='Resumen', index=False)
        
        # Hoja 2: Logs por época de cada experimento
        logs_data = []
        for nombre, metricas in resultados.items():
            for log in metricas['epoch_logs']:
                row = {
                    'Experimento': nombre,
                    'Época': log['epoch'],
                    'Loss': round(log['loss'], 4)
                }
                
                # Agregar accuracy o perplexity según el tipo
                if 'accuracy' in log:
                    row['Accuracy'] = round(log['accuracy'], 4)
                    row['Perplexity'] = None
                else:
                    row['Accuracy'] = None
                    row['Perplexity'] = round(log['perplexity'], 2)
                
                logs_data.append(row)
        
        df_logs = pd.DataFrame(logs_data)
        df_logs.to_excel(writer, sheet_name='Logs por Época', index=False)
        
        # Hoja 3: Comparativa (solo métricas finales)
        comparativa_data = []
        for nombre, metricas in resultados.items():
            comparativa_data.append({
                'Experimento': nombre,
                'Tiempo Total (s)': round(metricas['tiempo'], 2),
                'Throughput': round(metricas['throughput'], 2),
                'Métrica Final': round(metricas['accuracy'], 4),
                'Tipo': 'Perplexity' if 'WikiText' in nombre else 'Accuracy'
            })
        
        df_comparativa = pd.DataFrame(comparativa_data)
        df_comparativa.to_excel(writer, sheet_name='Comparativa', index=False)
        
        # Hoja 4: Detalles por experimento (cada experimento tiene sus épocas)
        for nombre, metricas in resultados.items():
            # Nombre de hoja válido (Excel tiene límite de 31 caracteres)
            sheet_name = nombre[:31]
            
            epoca_data = []
            for log in metricas['epoch_logs']:
                row = {
                    'Época': log['epoch'],
                    'Loss': round(log['loss'], 4)
                }
                
                if 'accuracy' in log:
                    row['Accuracy'] = round(log['accuracy'], 4)
                else:
                    row['Perplexity'] = round(log['perplexity'], 2)
                
                epoca_data.append(row)
            
            df_experimento = pd.DataFrame(epoca_data)
            
            # Agregar información del experimento al principio
            info_row = pd.DataFrame([{
                'Época': 'INFO',
                'Loss': f"Tiempo: {metricas['tiempo']:.2f}s",
                **({'Accuracy': f"Throughput: {metricas['throughput']:.2f}"} if 'accuracy' in metricas['epoch_logs'][0] 
                   else {'Perplexity': f"Throughput: {metricas['throughput']:.2f}"})
            }])
            
            df_final = pd.concat([info_row, df_experimento], ignore_index=True)
            df_final.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"📊 Excel guardado en: {excel_filename}")
    
    # ========== GUARDAR RESUMEN EN TXT ==========
    txt_filename = base_filename + ".txt"
    
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RESUMEN DE EXPERIMENTOS\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for nombre, metricas in resultados.items():
            f.write(f"\n{nombre}:\n")
            f.write(f"  Tiempo: {metricas['tiempo']:.2f}s\n")
            f.write(f"  Throughput: {metricas['throughput']:.2f} samples/s\n")
            if 'WikiText' in nombre:
                f.write(f"  Mejor Perplexity: {metricas['accuracy']:.2f}\n")
            else:
                f.write(f"  Mejor Accuracy: {metricas['accuracy']:.4f}\n")
            
            f.write("\n  Logs por época:\n")
            for log in metricas['epoch_logs']:
                f.write(f"    Época {log['epoch']}: Loss={log['loss']:.4f}")
                if 'accuracy' in log:
                    f.write(f", Acc={log['accuracy']:.4f}\n")
                else:
                    f.write(f", PPL={log['perplexity']:.2f}\n")
    
    print(f"📄 Resumen TXT guardado en: {txt_filename}")
    print(f"\n✅ Todos los archivos guardados en la carpeta 'results/'")


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
    
    resultados = {}
    
    # EXPERIMENTO 1: MNIST + MLP
    print("\n" + "="*70)
    print("EXPERIMENTO 1: MNIST + MLP")
    print("="*70)
    train_loader, test_loader = get_mnist(batch_size=64)
    model = MLP()
    resultados['MNIST + MLP'] = train_model(
        model, train_loader, test_loader, 
        epochs=5, device=device, name="MNIST + MLP"
    )
    
    # EXPERIMENTO 2: MNIST + CNN Small
    print("\n" + "="*70)
    print("EXPERIMENTO 2: MNIST + CNN Small")
    print("="*70)
    train_loader, test_loader = get_mnist(batch_size=64)
    model = SmallCNN_MNIST()
    resultados['MNIST + CNN Small'] = train_model(
        model, train_loader, test_loader,
        epochs=5, device=device, name="MNIST + CNN Small"
    )
    
    # EXPERIMENTO 3: CIFAR-10 + CNN Medium
    print("\n" + "="*70)
    print("EXPERIMENTO 3: CIFAR-10 + CNN Medium")
    print("="*70)
    train_loader, test_loader = get_cifar10(batch_size=64)
    model = MediumCNN()
    resultados['CIFAR-10 + CNN Medium'] = train_model(
        model, train_loader, test_loader,
        epochs=5, device=device, name="CIFAR-10 + CNN Medium"
    )
    
    # EXPERIMENTO 4: CIFAR-100 + CNN Large
    print("\n" + "="*70)
    print("EXPERIMENTO 4: CIFAR-100 + CNN Large")
    print("="*70)
    train_loader, test_loader = get_cifar100(batch_size=64)
    model = LargeCNN(num_classes=100)
    resultados['CIFAR-100 + CNN Large'] = train_model(
        model, train_loader, test_loader,
        epochs=10, device=device, name="CIFAR-100 + CNN Large"
    )
    
    # EXPERIMENTO 5: IMDB + Transformer Small
    print("\n" + "="*70)
    print("EXPERIMENTO 5: IMDB + Transformer Small")
    print("="*70)
    train_loader, test_loader, vocab_size = get_imdb(batch_size=32)
    model = SmallTransformer(vocab_size=vocab_size)
    resultados['IMDB + Transformer Small'] = train_model(
        model, train_loader, test_loader,
        epochs=3, device=device, 
        name="IMDB + Transformer Small",
        is_transformer=True
    )
    
    # EXPERIMENTO 6: WikiText-103 + Transformer Medium
    print("\n" + "="*70)
    print("EXPERIMENTO 6: WikiText-103 + Transformer Medium")
    print("="*70)
    train_loader, test_loader, vocab_size = get_wikitext103(batch_size=16, subset_size=5000)
    model = MediumTransformer(vocab_size=vocab_size)
    resultados['WikiText + Transformer Medium'] = train_model(
        model, train_loader, test_loader,
        epochs=3, device=device,
        name="WikiText + Transformer Medium",
        is_transformer=True, is_lm=True
    )
    
    # RESUMEN FINAL
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
    
    # GUARDAR RESULTADOS
    save_results(resultados)
    
    return resultados


if __name__ == "__main__":
    resultados = main()