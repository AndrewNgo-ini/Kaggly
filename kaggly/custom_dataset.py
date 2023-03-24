class custom_dataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], tokenizer)
        label = torch.tensor(self.labels[item])
        inputs["labels"] = label
        return inputs