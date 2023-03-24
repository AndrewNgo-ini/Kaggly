import torch
from torch.utils.data import Dataset

def prepare_input(text, tokenizer):
    inputs = tokenizer.encode_plus(
        text, 
        return_tensors = None, 
        add_special_tokens = True, 
        truncation = True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

class KagglyDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.tokenizer)
        label = torch.tensor(self.labels[item])
        inputs["labels"] = label
        return inputs