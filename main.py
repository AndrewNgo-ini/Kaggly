from cfg import CFG

if CFG.use_wandb:
    
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    key = user_secrets.get_secret("wandb")

    
    import wandb
    wandb.login(key=key)

for fold in range(CFG.folds):

    print(f" Starting fold {fold} ".center(30, "*"))
    !accelerate launch train.py --fold $fold
    print("\n\n")