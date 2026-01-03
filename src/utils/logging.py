"""
Utilidades para logging y visualización de entrenamiento
"""


def print_experiment_header(name, num_params, lr, epochs):
    """Imprime el encabezado de un experimento."""
    print(f"\n{'='*70}")
    print(f"🚀 {name}")
    print(f"   Parámetros: {num_params:,}")
    print(f"   LR: {lr} | Épocas: {epochs}")
    print(f"{'='*70}")


def print_epoch_progress(epoch, total_epochs, batch_idx, total_batches, loss, is_lm=False, ppl=None):
    """Imprime el progreso durante una época."""
    if is_lm and ppl is not None:
        print(f"  [Epoch {epoch+1}/{total_epochs}] Batch {batch_idx}/{total_batches} | "
              f"Loss: {loss:.4f} | PPL: {ppl:.2f}")
    else:
        print(f"  [Epoch {epoch+1}/{total_epochs}] Batch {batch_idx}/{total_batches} | "
              f"Loss: {loss:.4f}")


def print_epoch_summary(epoch, total_epochs, avg_loss, metric_value, best_metric, is_lm=False):
    """Imprime el resumen de una época."""
    if is_lm:
        print(f"  ✓ [Epoch {epoch+1}/{total_epochs}] Loss: {avg_loss:.4f} | "
              f"Perplexity: {metric_value:.2f} | Best: {best_metric:.2f}")
    else:
        print(f"  ✓ [Epoch {epoch+1}/{total_epochs}] Loss: {avg_loss:.4f} | "
              f"Acc: {metric_value:.4f} | Best: {best_metric:.4f}")


def print_training_summary(elapsed_time, throughput, best_metric, is_lm=False):
    """Imprime el resumen final del entrenamiento."""
    print(f"\n✅ Completado | Tiempo: {elapsed_time:.2f}s | "
          f"Throughput: {throughput:.2f} samples/s")
    if is_lm:
        print(f"🏆 Mejor Perplexity: {best_metric:.2f}")
    else:
        print(f"🏆 Mejor Accuracy: {best_metric:.4f}")


def print_best_model_update(best_metric, is_lm=False):
    """Imprime cuando se encuentra un nuevo mejor modelo."""
    if is_lm:
        print(f"   ⭐ Mejor perplexity: {best_metric:.2f}")
    else:
        print(f"   ⭐ Mejor accuracy: {best_metric:.4f}")