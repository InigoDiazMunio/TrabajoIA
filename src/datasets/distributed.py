"""
Dataset loaders con DistributedSampler para DDP
"""
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from datasets import load_dataset
from transformers import AutoTokenizer
from .imdb import IMDBDataset
from .wikitext import WikiTextDataset


def get_mnist_distributed(batch_size, rank, world_size, data_dir='./data'):
    """
    Carga MNIST con DistributedSampler para DDP.
    
    Args:
        batch_size: Tamaño del batch
        rank: Rank del proceso actual
        world_size: Número total de procesos
        data_dir: Directorio de datos
        
    Returns:
        train_loader, test_loader, train_sampler
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, 
                            num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_sampler


def get_cifar10_distributed(batch_size, rank, world_size, data_dir='./data'):
    """Carga CIFAR-10 con DistributedSampler para DDP."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                            num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_sampler


def get_cifar100_distributed(batch_size, rank, world_size, data_dir='./data'):
    """Carga CIFAR-100 con DistributedSampler para DDP."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                            num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_sampler


def get_imdb_distributed(batch_size, rank, world_size, max_length=128):
    """Carga IMDB con DistributedSampler para DDP."""
    if rank == 0:
        print("Descargando IMDB...")
    
    dataset = load_dataset('imdb')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = IMDBDataset(dataset['train']['text'], dataset['train']['label'], tokenizer, max_length)
    test_dataset = IMDBDataset(dataset['test']['text'], dataset['test']['label'], tokenizer, max_length)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                            num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_sampler


def get_wikitext103_distributed(batch_size, rank, world_size, max_length=256, subset_size=5000):
    """Carga WikiText-103 con DistributedSampler para DDP."""
    if rank == 0:
        print(f"Descargando WikiText-103 (usando {subset_size} muestras)...")
    
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_texts = dataset['train']['text'][:subset_size]
    test_texts = dataset['test']['text'][:subset_size // 10]

    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length)
    test_dataset = WikiTextDataset(test_texts, tokenizer, max_length)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                            num_workers=2, pin_memory=True)

    return train_loader, test_loader, train_sampler