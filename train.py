#%%writefile train.py


import gc
import json
from pathlib import Path
from itertools import chain

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    set_seed,
    get_scheduler,
)
from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs

from modeling import SBert, MultipleNegativesRankingLoss
from utils import compute_metrics
from cfg import CFG
from metric import RecallAtK
from collator import MNRCollator


def get_pos_and_tt_ids(input_ids, device):
    """
    This is a weird thing needed when using a distributed setup.
    If you don't have it, you face this: https://github.com/huggingface/accelerate/issues/97
    """
    position_ids = torch.arange(0, input_ids.shape[-1], device=device).expand((1, -1))
    token_type_ids = torch.zeros(
                input_ids.shape,
                dtype=torch.long,
                device=position_ids.device,
            )
    
    return {
        "position_ids": position_ids, 
        "token_type_ids": token_type_ids
    }
    

def main(fold):
    
    output_dir = Path(f"{CFG.output_dir}-fold{fold}")
    output_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    with open("training_idxs.json") as fp:
        fold_idxs = json.load(fp)

    tokenized_ds = load_dataset(
        "parquet", data_files=CFG.tokenized_ds_name, split="train"
    )
    combined = pd.read_parquet("combined2.pq")

    set_seed(CFG.seed)

    accelerator = Accelerator(
        mixed_precision=CFG.mixed_precision,
        project_dir=CFG.output_dir,
        log_with="wandb" if CFG.use_wandb else None,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)] # Another weird workaround for this: https://github.com/huggingface/accelerate/issues/648
    )

    if CFG.use_wandb:
        config = {k: getattr(CFG, k) for k in dir(CFG) if not k.startswith("_")}
        accelerator.init_trackers(CFG.output_dir, config)

    train_idxs = list(chain(*[x for i, x in fold_idxs.items() if i != str(fold)]))
    train_ds = tokenized_ds.select(train_idxs)
    val_ds = tokenized_ds.select(fold_idxs[str(fold)])
    val_cols = ["topic_id", "content_id", "topic_language", "content_language"]
    val_df = combined.loc[fold_idxs[str(fold)], val_cols].reset_index(drop=True).copy()
    
    del combined, tokenized_ds
    gc.collect()
    
    accelerator.print(f"{len(train_ds)} training examples, {len(val_ds)} validation examples")

    metric = RecallAtK(val_df, k=[10, 50, 100])

    n_train_steps = int(
        len(train_ds) / CFG.batch_size / CFG.grad_accum * CFG.epochs / CFG.num_devices
    )
    
    epoch_steps = int(n_train_steps / CFG.epochs)
    log_steps = int(epoch_steps/CFG.log_per_epoch)
    eval_steps = int(epoch_steps/CFG.evals_per_epoch)

    data_collator = MNRCollator(
        tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )

    train_dataloader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=CFG.batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        val_ds, 
        collate_fn=data_collator, 
        batch_size=CFG.batch_size * 2,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    with accelerator.main_process_first():
        model = SBert(CFG.model_name)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": CFG.wd,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=CFG.lr)

    lr_scheduler = get_scheduler(
        name=CFG.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(n_train_steps * CFG.warmup_ratio),
        num_training_steps=n_train_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )
    
    loss_fct = MultipleNegativesRankingLoss()

    completed_steps = 0
    last_logged_global_step = 0
    global_step = 0
    progress_bar = tqdm(range(n_train_steps), disable=not accelerator.is_local_main_process)
    best_score = 0
    
    
    def eval_loop():
        model.eval()
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, disable=not accelerator.is_local_main_process):
                embeds_a = model(
                    batch["input_ids_a"], 
                    batch["attention_mask_a"],
                    **get_pos_and_tt_ids(batch["input_ids_a"], accelerator.device),
                ).pooled_embeddings
                embeds_b = model(
                    batch["input_ids_b"], 
                    batch["attention_mask_b"],
                    **get_pos_and_tt_ids(batch["input_ids_b"], accelerator.device)
                ).pooled_embeddings

                all_embeds_a, all_embeds_b = accelerator.gather_for_metrics(
                    (embeds_a, embeds_b)
                )

                metric.add_batch(predictions_a=all_embeds_a, predictions_b=all_embeds_b)

        return metric.compute()
    
    
    untrained_scores = eval_loop()
    accelerator.print(f"Before training | Eval scores {untrained_scores}")
    
    if CFG.use_wandb:
        accelerator.log(
            {
                **{f"validation/{k}": v for k, v in untrained_scores.items()},
            },
            step=global_step,
        )
    
    
    for epoch in range(CFG.epochs):
        model.train()

        if CFG.use_wandb:
            train_loss = torch.tensor(0.0, device=accelerator.device)

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            embeddings_a = model(
                batch["input_ids_a"],
                batch["attention_mask_a"],
                **get_pos_and_tt_ids(batch["input_ids_a"], accelerator.device)
            ).pooled_embeddings
            embeddings_b = model(
                batch["input_ids_b"],
                batch["attention_mask_b"],
                **get_pos_and_tt_ids(batch["input_ids_b"], accelerator.device)
            ).pooled_embeddings

            loss = loss_fct(embeddings_a, embeddings_b)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            if CFG.use_wandb:
                train_loss += loss

            if (step+1) % log_steps == 0:
                
                tr_loss = round(accelerator.gather(train_loss).mean().item()/(global_step-last_logged_global_step), 4)
                last_logged_global_step = global_step
                train_loss -= train_loss
                
                accelerator.log({
                    "training/loss": tr_loss,
                },
                    step=global_step
                )
                accelerator.print(f"Epoch {epoch+round(step/epoch_steps, 2)}/{CFG.epochs} | Loss {tr_loss}")
                
            if (step+1) % eval_steps == 0:
                scores = eval_loop()
                accelerator.print(f"Epoch {round(epoch+step/epoch_steps, 2)}/{CFG.epochs} | Eval scores {scores}")
                
                metric_score = scores[CFG.metric_to_track]
                
                if metric_score > best_score:
                    accelerator.print(f"New best {CFG.metric_to_track}: {round(metric_score, 4)} \nSaving model")
                    state_dict = accelerator.get_state_dict(model)
                    accelerator.save(state_dict, str(output_dir/"model_state_dict.pt"))
                    model
                    
                    best_score = metric_score
                    
                
                model.train()

                if CFG.use_wandb:
                    accelerator.log(
                        {
                            **{f"validation/{k}": v for k, v in scores.items()},
                        },
                        step=global_step,
                    )
                    
                    
    if CFG.use_wandb:
        accelerator.end_training()
    
    accelerator.free_memory()
    
    del metric
    gc.collect()

if __name__ == "__main__":
    import fire
    # fire makes it really easy to pass args
    fire.Fire(main)