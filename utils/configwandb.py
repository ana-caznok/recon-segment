from transforms.factory import transform_factory

def config2wandb(config):
    config_wandb = {} 
    for k in config: 
        if (k == 'train') or (k=='valid'): 
            for l in config[k]:
                config_wandb[k+'_'+l] = config[k][l]
        else: 
            config_wandb[k] = config[k]
    string_config = transform_factory(config['train']['transform_index']).__str__().replace('\n',' ')
    config_wandb['transform_steps'] = string_config
    return config_wandb