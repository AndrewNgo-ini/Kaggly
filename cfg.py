#%%writefile cfg.py

import torch

class CFG:

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size = 256
    output_dir = model_name.split("/")[-1] + "-MNR-tuned"

    folds = 4
    lr = 5e-5
    wd = 0.01
    warmup_ratio = 0.1
    epochs = 2
    evals_per_epoch = 2
    log_per_epoch = 20  
    grad_accum = 1
    num_devices = torch.cuda.device_count()
    scheduler_type = "cosine"
    mixed_precision = "fp16"

    topic_cols = ["title", "description"]
    content_cols = ["title", "description", "text"]
    max_length = 128
    num_proc = 4
    
    tokenized_ds_name = "tokenized.pq"
    use_wandb = True
    debug = False
    seed = 18
    
    metric_to_track = "recall@100"