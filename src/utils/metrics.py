"""
Utilidades para métricas y seguimiento de entrenamiento
"""
import torch


def count_params(model):
    """Cuenta el número de parámetros entrenables en un modelo."""
    if hasattr(model, 'module'):
        model = model.module
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_accuracy(outputs, labels):
    """Calcula accuracy para clasificación."""
    pred = outputs.argmax(dim=1)
    correct = (pred == labels).sum().item()
    total = labels.size(0)
    return correct / total


def calculate_perplexity(loss):
    """Calcula perplexity a partir del loss."""
    return torch.exp(loss).item()