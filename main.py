import yaml
import os
from pathlib import Path

d = {
    "command_file": None,
    "commands": None,
    "compute_environment": "LOCAL_MACHINE",
    "deepspeed_config": {},
    "distributed_type": "MULTI_GPU", # choose from {"NO", "MULTI_GPU"}
    "downcast_bf16": "no",
    "dynamo_backend": "NO",
    "fsdp_config": {},
    "gpu_ids": "all",
    "machine_rank": 0,
    "main_process_ip": None,
    "main_process_port": None,
    "main_training_function": "main",
    "megatron_lm_config": {},
    "mixed_precision": "fp16", # choose from {"no", "bf16", "fp16"}
    "num_machines": 1,
    "num_processes": 2, # number of gpus
    "rdzv_backend": "static",
    "same_network": True,
    "tpu_name": None,
    "tpu_zone": None,
    "use_cpu": False,
}



config_dir = Path.home() / ".cache/huggingface/accelerate"
    
config_dir.mkdir(exist_ok=True, parents=True)

with open(config_dir / "default_config.yaml", "w") as fp:
    yaml.dump(d, fp)
    
#!cat ~/.cache/huggingface/accelerate/default_config.yaml

from cfg import CFG
import os
if CFG.use_wandb:
    
    # from kaggle_secrets import UserSecretsClient
    # user_secrets = UserSecretsClient()
    # key = user_secrets.get_secret("wandb")

    
    import wandb
    wandb.login(key="e354c084c06711db1fb03871f613a8962f692985")

for fold in range(CFG.folds):

    print(f" Starting fold {fold} ".center(30, "*"))
    os.system("accelerate launch train.py --fold $fold")
    print("\n\n")