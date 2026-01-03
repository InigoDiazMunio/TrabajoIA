"""
Dataset loader para IMDB (clasificación de sentimientos)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class IMDBDataset(Dataset):
    """Dataset personalizado para IMDB con tokenización"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def get_imdb(batch_size, max_length=128):
    """Carga el dataset IMDB para clasificación de sentimientos."""
    print("Descargando IMDB...")
    dataset = load_dataset('imdb')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = IMDBDataset(dataset['train']['text'], dataset['train']['label'], tokenizer, max_length)
    test_dataset = IMDBDataset(dataset['test']['text'], dataset['test']['label'], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, tokenizer.vocab_size