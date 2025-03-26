def config2wandb(config):
    config_wandb = {} 
    for k in config: 
        if (k == 'train') or (k=='valid'): 
            for l in config[k]:
                config_wandb[k+'_'+l] = config[k][l]
        else: 
            config_wandb[k] = config[k]
    return config_wandb