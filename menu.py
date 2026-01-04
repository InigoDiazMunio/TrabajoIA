"""
Script maestro para ejecutar experimentos con diferentes métodos de paralelización.
Menú interactivo para elegir qué método ejecutar.
"""

import sys
import os
import torch

def mostrar_menu():
    """Muestra el menú principal."""
    print("\n" + "="*70)
    print("  ENTRENAMIENTO DISTRIBUIDO - MENÚ PRINCIPAL")
    print("="*70)
    print("\nMétodos disponibles:")
    print("  1. Baseline (Single GPU)")
    print("  2. DataParallel (Multi-GPU)")
    print("  3. DistributedDataParallel / DDP (Multi-GPU Distribuido)")
    print("  4. Ejecutar TODOS los métodos (secuencialmente)")
    print("  5. Información del sistema")
    print("  0. Salir")
    print("\n" + "="*70)


def mostrar_info_sistema():
    """Muestra información del sistema."""
    print("\n" + "="*70)
    print("  INFORMACIÓN DEL SISTEMA")
    print("="*70)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Nombre: {torch.cuda.get_device_name(i)}")
            print(f"  Memoria: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    else:
        print("\n⚠️  No hay GPUs disponibles. Los entrenamientos serán en CPU (muy lentos).")
    
    print("\n" + "="*70)


def ejecutar_baseline():
    """Ejecuta el método baseline."""
    print("\n" + "="*70)
    print("  EJECUTANDO: Baseline (Single GPU)")
    print("="*70 + "\n")
    
    os.system("python run.py")
    
    print("\n✅ Baseline completado")
    input("\nPresiona Enter para continuar...")


def ejecutar_dataparallel():
    """Ejecuta el método DataParallel."""
    num_gpus = torch.cuda.device_count()
    
    print("\n" + "="*70)
    print("  EJECUTANDO: DataParallel (Multi-GPU)")
    print("="*70)
    print(f"\nGPUs disponibles: {num_gpus}")
    
    if num_gpus < 2:
        print("\n⚠️  ADVERTENCIA: Solo hay 1 GPU disponible.")
        print("   DataParallel funcionará pero no habrá speedup real.")
        respuesta = input("\n¿Deseas continuar? (s/n): ")
        if respuesta.lower() != 's':
            return
    
    print()
    os.system("python run_dataparallel.py")
    
    print("\n✅ DataParallel completado")
    input("\nPresiona Enter para continuar...")


def ejecutar_ddp():
    """Ejecuta el método DDP."""
    num_gpus = torch.cuda.device_count()
    
    print("\n" + "="*70)
    print("  EJECUTANDO: DistributedDataParallel (DDP)")
    print("="*70)
    print(f"\nGPUs disponibles: {num_gpus}")
    
    if num_gpus < 2:
        print("\n⚠️  ADVERTENCIA CRÍTICA: DDP requiere al menos 2 GPUs.")
        print("   Con 1 GPU, DDP no funcionará correctamente.")
        respuesta = input("\n¿Deseas continuar de todos modos? (s/n): ")
        if respuesta.lower() != 's':
            return
    
    print()
    os.system("python run_ddp.py")
    
    print("\n✅ DDP completado")
    input("\nPresiona Enter para continuar...")


def ejecutar_todos():
    """Ejecuta todos los métodos secuencialmente."""
    print("\n" + "="*70)
    print("  EJECUTANDO: TODOS LOS MÉTODOS")
    print("="*70)
    print("\nSe ejecutarán en orden:")
    print("  1. Baseline")
    print("  2. DataParallel")
    print("  3. DDP")
    print("\nEsto puede tomar varias horas.")
    
    respuesta = input("\n¿Estás seguro de continuar? (s/n): ")
    if respuesta.lower() != 's':
        return
    
    # Ejecutar baseline
    print("\n" + "="*70)
    print("  [1/3] Ejecutando Baseline...")
    print("="*70 + "\n")
    os.system("python run.py")
    print("\n✅ Baseline completado")
    
    # Ejecutar DataParallel
    print("\n" + "="*70)
    print("  [2/3] Ejecutando DataParallel...")
    print("="*70 + "\n")
    os.system("python run_dataparallel.py")
    print("\n✅ DataParallel completado")
    
    # Ejecutar DDP
    print("\n" + "="*70)
    print("  [3/3] Ejecutando DDP...")
    print("="*70 + "\n")
    os.system("python run_ddp.py")
    print("\n✅ DDP completado")
    
    print("\n" + "="*70)
    print("  ✅ TODOS LOS MÉTODOS COMPLETADOS")
    print("="*70)
    print("\nResultados guardados en la carpeta 'results/'")
    print("Abre los archivos Excel para comparar los resultados.")
    
    input("\nPresiona Enter para continuar...")


def main():
    """Función principal del menú."""
    while True:
        mostrar_menu()
        
        try:
            opcion = input("\nSelecciona una opción (0-5): ").strip()
            
            if opcion == '0':
                print("\n👋 Saliendo del programa. ¡Hasta luego!")
                break
            
            elif opcion == '1':
                ejecutar_baseline()
            
            elif opcion == '2':
                ejecutar_dataparallel()
            
            elif opcion == '3':
                ejecutar_ddp()
            
            elif opcion == '4':
                ejecutar_todos()
            
            elif opcion == '5':
                mostrar_info_sistema()
                input("\nPresiona Enter para continuar...")
            
            else:
                print("\n❌ Opción inválida. Por favor selecciona 0-5.")
                input("\nPresiona Enter para continuar...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrumpido. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TRABAJO FIN DE GRADO - ENTRENAMIENTO DISTRIBUIDO")
    print("  Iñigo - Universidad del País Vasco (UPV/EHU)")
    print("="*70)
    
    main()