"""
Módulo de entrenamiento para DataParallel (nn.DataParallel)
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils import (
    count_params,
    print_experiment_header,
    print_epoch_progress,
    print_epoch_summary,
    print_training_summary,
    print_best_model_update
)


def train_model_dataparallel(model, train_loader, test_loader, epochs, device, name,
                             is_transformer=False, is_lm=False):
    """
    Entrena un modelo usando DataParallel (nn.DataParallel) para multi-GPU.
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        test_loader: DataLoader de test
        epochs: Número de épocas
        device: Dispositivo principal ('cuda:0')
        name: Nombre del experimento
        is_transformer: Si el modelo es un Transformer
        is_lm: Si es una tarea de language modeling
        
    Returns:
        dict: Diccionario con métricas del entrenamiento + memoria GPU
    """
    num_gpus = torch.cuda.device_count()
    
    # Reset memoria GPU
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated() / 1e9
    
    # Mover modelo a GPU principal
    model = model.to(device)
    
    # Envolver con DataParallel si hay múltiples GPUs
    if num_gpus > 1:
        print(f"\n Usando DataParallel con {num_gpus} GPUs")
        model = nn.DataParallel(model)
    else:
        print(f"\n  Solo 1 GPU disponible - DataParallel no tendrá speedup")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0 if is_lm else -100)

    # Learning rate adaptativo
    if is_transformer:
        lr = 2e-5
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        lr = 1e-3
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    # Scheduler para CNNs grandes
    if not is_transformer:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    print_experiment_header(name, count_params(model), lr, epochs)
    print(f"   GPUs: {num_gpus} | Memoria inicial: {start_memory:.2f} GB")

    start_time = time.time()
    epoch_logs = []
    best_metric = 0.0 if not is_lm else float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

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

            # Gradient clipping para transformers
            if is_transformer or is_lm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            # Imprimir progreso cada 100 batches
            if batch_idx % 100 == 0:
                if is_lm:
                    ppl = torch.exp(loss).item()
                    print_epoch_progress(epoch, epochs, batch_idx, len(train_loader),
                                       loss.item(), is_lm=True, ppl=ppl)
                else:
                    print_epoch_progress(epoch, epochs, batch_idx, len(train_loader),
                                       loss.item())

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

        # Calcular métrica según tipo de tarea
        if is_lm:
            perplexity = torch.exp(torch.tensor(eval_loss / eval_batches)).item()
            metric_value = perplexity

            if perplexity < best_metric:
                best_metric = perplexity
                print_best_model_update(best_metric, is_lm=True)

            epoch_logs.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'perplexity': perplexity
            })

            print_epoch_summary(epoch, epochs, avg_loss, perplexity, best_metric, is_lm=True)

        else:
            accuracy = correct / total
            metric_value = accuracy

            if not is_transformer:
                scheduler.step(accuracy)

            if accuracy > best_metric:
                best_metric = accuracy
                print_best_model_update(best_metric)

            epoch_logs.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy
            })

            print_epoch_summary(epoch, epochs, avg_loss, accuracy, best_metric)

    elapsed = time.time() - start_time
    throughput = len(train_loader.dataset) * epochs / elapsed
    
    # Obtener memoria GPU pico
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    print_training_summary(elapsed, throughput, best_metric, is_lm)
    print(f" Memoria GPU pico: {peak_memory:.2f} GB")

    return {
        'tiempo': elapsed,
        'throughput': throughput,
        'accuracy': best_metric,
        'memoria_gb': peak_memory,
        'num_gpus': num_gpus,
        'epoch_logs': epoch_logs
    }