"""
Módulo de entrenamiento para DistributedDataParallel (DDP)
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils import count_params


def setup_ddp(rank, world_size):
    """
    Inicializa el proceso group para DDP.
    
    Args:
        rank: Rank del proceso actual
        world_size: Número total de procesos (GPUs)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Usar NCCL en Linux/GPU, GLOO en Windows
    backend = 'nccl' if torch.cuda.is_available() and os.name != 'nt' else 'gloo'
    
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Limpia el proceso group de DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Retorna True si es el proceso principal (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def train_ddp_worker(rank, world_size, model_fn, train_loader_fn, test_loader_fn,
                     epochs, name, is_transformer=False, is_lm=False):
    """
    Función worker para entrenar con DDP en cada GPU.
    
    Args:
        rank: Rank de este proceso/GPU
        world_size: Número total de GPUs
        model_fn: Función que retorna el modelo
        train_loader_fn: Función que retorna train_loader para este rank
        test_loader_fn: Función que retorna test_loader para este rank
        epochs: Número de épocas
        name: Nombre del experimento
        is_transformer: Si el modelo es un Transformer
        is_lm: Si es language modeling
        
    Returns:
        dict: Métricas (solo desde rank 0)
    """
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Tracking de memoria GPU
    torch.cuda.reset_peak_memory_stats(device)
    start_memory = torch.cuda.memory_allocated(device) / 1e9
    
    if is_main_process():
        print(f"\n{'='*70}")
        print(f"🚀 DDP: {name} | GPUs: {world_size}")
        print(f"   Memoria inicial: {start_memory:.2f} GB")
        print(f"{'='*70}")
    
    # Crear modelo y moverlo a la GPU correspondiente
    model = model_fn()
    model = model.to(device)
    
    # Envolver con DDP
    model = DDP(model, device_ids=[rank])
    
    if is_main_process():
        print(f"   Parámetros: {count_params(model):,}")
    
    # Obtener data loaders
    train_loader = train_loader_fn(rank, world_size)
    test_loader = test_loader_fn(rank, world_size)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0 if is_lm else -100)
    
    # Learning rate adaptativo
    if is_transformer:
        lr = 2e-5
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        lr = 1e-3
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # Scheduler para CNNs
    if not is_transformer:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    start_time = time.time()
    epoch_logs = []
    best_metric = 0.0 if not is_lm else float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Importante: set_epoch para shuffle correcto
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            if is_transformer or is_lm:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                
                if is_lm:
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                else:
                    loss = criterion(outputs, labels)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
            
            loss.backward()
            
            # Gradient clipping
            if is_transformer or is_lm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Imprimir progreso (solo rank 0)
            if is_main_process() and batch_idx % 100 == 0:
                if is_lm:
                    ppl = torch.exp(loss).item()
                    print(f"  [Epoch {epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | PPL: {ppl:.2f}")
                else:
                    print(f"  [Epoch {epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f}")
        
        # Evaluar
        model.eval()
        correct, total = 0, 0
        eval_loss = 0.0
        eval_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if is_transformer or is_lm:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    outputs = model(input_ids, attention_mask)
                    
                    if is_lm:
                        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                        eval_loss += loss.item()
                        eval_batches += 1
                    else:
                        pred = outputs.argmax(dim=1)
                        correct += (pred == labels).sum().item()
                        total += labels.size(0)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    pred = outputs.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
        
        avg_loss = train_loss / len(train_loader)
        
        # Sincronizar métricas entre procesos
        if is_lm:
            eval_loss_tensor = torch.tensor(eval_loss).to(device)
            eval_batches_tensor = torch.tensor(eval_batches).to(device)
            dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_batches_tensor, op=dist.ReduceOp.SUM)
            
            perplexity = torch.exp(eval_loss_tensor / eval_batches_tensor).item()
            metric_value = perplexity
            
            if perplexity < best_metric:
                best_metric = perplexity
                if is_main_process():
                    print(f"   ⭐ Mejor perplexity: {best_metric:.2f}")
            
            if is_main_process():
                epoch_logs.append({
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'perplexity': perplexity
                })
                print(f"  ✓ [Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | "
                      f"Perplexity: {perplexity:.2f} | Best: {best_metric:.2f}")
        else:
            correct_tensor = torch.tensor(correct).to(device)
            total_tensor = torch.tensor(total).to(device)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            
            accuracy = (correct_tensor / total_tensor).item()
            metric_value = accuracy
            
            if not is_transformer:
                scheduler.step(accuracy)
            
            if accuracy > best_metric:
                best_metric = accuracy
                if is_main_process():
                    print(f"   ⭐ Mejor accuracy: {best_metric:.4f}")
            
            if is_main_process():
                epoch_logs.append({
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'accuracy': accuracy
                })
                print(f"  ✓ [Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | "
                      f"Acc: {accuracy:.4f} | Best: {best_metric:.4f}")
    
    elapsed = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
    
    if is_main_process():
        throughput = len(train_loader.dataset) * world_size * epochs / elapsed
        print(f"\n✅ Completado | Tiempo: {elapsed:.2f}s | "
              f"Throughput: {throughput:.2f} samples/s")
        if is_lm:
            print(f"🏆 Mejor Perplexity: {best_metric:.2f}")
        else:
            print(f"🏆 Mejor Accuracy: {best_metric:.4f}")
        print(f"💾 Memoria GPU pico: {peak_memory:.2f} GB")
    
    cleanup()
    
    # Solo rank 0 retorna resultados
    if rank == 0:
        return {
            'tiempo': elapsed,
            'throughput': throughput,
            'accuracy': best_metric,
            'memoria_gb': peak_memory,
            'num_gpus': world_size,
            'epoch_logs': epoch_logs
        }
    return None


def train_model_ddp(model_fn, train_loader_fn, test_loader_fn, epochs, name,
                    is_transformer=False, is_lm=False):
    """
    Lanza el entrenamiento DDP en múltiples GPUs.
    
    Args:
        model_fn: Función que crea y retorna el modelo
        train_loader_fn: Función(rank, world_size) que retorna train_loader
        test_loader_fn: Función(rank, world_size) que retorna test_loader
        epochs: Número de épocas
        name: Nombre del experimento
        is_transformer: Si es transformer
        is_lm: Si es language modeling
        
    Returns:
        dict: Métricas del entrenamiento
    """
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("⚠️  Solo 1 GPU disponible. DDP requiere al menos 2 GPUs.")
        print("   Usando single GPU en su lugar...")
        # Fallback a single GPU
        from .single_gpu import train_model
        model = model_fn()
        train_loader = train_loader_fn(0, 1)
        test_loader = test_loader_fn(0, 1)
        return train_model(model, train_loader, test_loader, epochs, 
                          torch.device('cuda:0'), name, is_transformer, is_lm)
    
    # Usar spawn para iniciar procesos
    mp.spawn(
        train_ddp_worker,
        args=(world_size, model_fn, train_loader_fn, test_loader_fn,
              epochs, name, is_transformer, is_lm),
        nprocs=world_size,
        join=True
    )
    
    # Leer resultados del rank 0
    return {'status': 'completed', 'num_gpus': world_size}