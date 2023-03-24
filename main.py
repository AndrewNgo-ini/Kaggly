

if __name__ == "__main__":
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    tokenizer= AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base",
                                                               num_labels=2, id2label=id2label, label2id=label2id)
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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ds = read_data_tok()
    lb = read_label_tok()

    random.seed(42)
    val_cluster = pick_subset_by_percentage(ds.keys(), 20)
    print(val_cluster)
    with open('val_cluster.pkl', 'wb') as file:
        pickle.dump(val_cluster, file)
    texts = []
    labels = []
    clusters = []
    for k,v in ds.items():
        for sentence in v:
            if sentence in lb[k]:
                labels.append(1)
            else:
                labels.append(0)
            texts.append(sentence)
            clusters.append(k)
    df = pd.DataFrame({"text": texts, "label": labels, "cluster": clusters})
    print(df.shape)
    print(df["label"].value_counts())

    ds_train = custom_dataset(df[~df["cluster"].isin(val_cluster)], tokenizer)
    ds_val = custom_dataset(df[df["cluster"].isin(val_cluster)], tokenizer)
    
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