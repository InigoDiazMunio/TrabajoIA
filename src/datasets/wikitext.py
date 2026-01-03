"""
Dataset loader para WikiText-103 (language modeling)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    """Dataset personalizado para WikiText con tokenización"""
    
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = [t for t in texts if len(t) > 10]
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
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        return input_ids, attention_mask, labels


def get_wikitext103(batch_size, max_length=256, subset_size=5000):
    """Carga el dataset WikiText-103 para language modeling."""
    print(f"Descargando WikiText-103 (usando {subset_size} muestras)...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_texts = dataset['train']['text'][:subset_size]
    test_texts = dataset['test']['text'][:subset_size // 10]

    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length)
    test_dataset = WikiTextDataset(test_texts, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, tokenizer.vocab_size