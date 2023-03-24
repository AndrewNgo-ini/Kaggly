from kaggly.model import KagglyModel
from kaggly.custom_dataset import KagglyDataset
from kaggly.metric import KagglyEvaluate
from transformers import TrainingArguments, Trainer

if __name__ == "__main__":
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = KagglyModel.from_pretrained("vinai/phobert-base", num_labels=2, id2label=id2label, label2id=label2id)
    # Freeze the pretrained BERT layers
    #for param in model.base_model.parameters():
    #    param.requires_grad = False

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    ds_train = KagglyDataset()
    ds_val = KagglyDataset()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    #trainer.save_model()