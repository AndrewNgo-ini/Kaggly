
from unicodedata import category
from src.data.make_dataset import prepare_data_and_label
import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets import Dataset, load_metric
import torch

category = "cohesion"
accuracy_metric = load_metric('accuracy')
ds = prepare_data_and_label(category)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=9)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



def score(preds): 
    return {'acc': accuracy_metric.compute(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}


training_args = TrainingArguments(
    output_dir="./models",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    #per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    #eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=score,
)

trainer.train()
trainer.save_model(category)