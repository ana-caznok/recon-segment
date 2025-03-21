from tqdm import trange, tqdm
import os
import sys
import math
import torch
import numpy as np
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tools_wandb import ToolsWandb
import yaml
from typing import Dict, Any
from utils import model_select, load_best_ckpt
from baseline.train_code.utils import AverageMeter, Loss_MRAE
from fixed_dataset import FixedDataset
from transforms import transform_factory
from torchmetrics.image import StructuralSimilarityIndexMeasure
from loss import loss_select
from metrics.sam import SAMScore
from evaluate import evaluation
from utils import fixed_save_checkpoint, read_yaml, config_flatten


RECORD_METRIC = None
CHANNEL_WEIGHT = [None]*61
CHANNEL_ERRORS = [None]*61

cw=[]
ce=[]

def save_cw(cw, fixed_checkpoint_name, save_checkpoint_path, tp): #new
    print('cw: ')
    print(cw)
    with open(save_checkpoint_path +'/'+ tp +'_'+ fixed_checkpoint_name+'.npy', 'wb') as f:
        np.save(f, np.array(cw))


def train(model: nn.Module, metrics: Dict[str, object], configs: Dict[str, Any], train_dataset: FixedDataset, val_dataset: FixedDataset, epochs: int, total_iteration: int, ep: int = 0, iter: int  = 0):
    iteration = iter
    fixed_checkpoint_name = configs["fixed_checkpoint_name"]

    # Getting loss param
    index = configs.get("loss", None)

    # Automatic Mixed Precision (AMP), needed for heavy 3D UNet
    # Ref: https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
    scaler = torch.cuda.amp.GradScaler()
    torch.set_float32_matmul_precision("high")

    # Added Fixed checkpoint option to avoid too many saved weights
    if not configs["debug"] and os.path.isfile(os.path.join(configs["save_checkpoint_path"], f"{fixed_checkpoint_name}.pth")):
        raise ValueError(f"Weight already exists! Use a different .yaml than {fixed_checkpoint_name} or delete/move previous weight.")

    with trange(epochs-ep, desc='Train Loop') as progress_bar:
        for epo_idx, epo in zip(progress_bar, range(ep, epochs + 1)):
            epoch_percentage = epo / (epochs + 1)
            model.train()
            losses = AverageMeter()
            nworkers = configs.get("nworkers", 1)
            if nworkers == "cpu_count":
                nworkers = os.cpu_count()
            train_loader = DataLoader(dataset=train_dataset, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=nworkers)
            val_loader = DataLoader(dataset=val_dataset, batch_size=configs['valid']['batch_size'], shuffle=False, num_workers=1)

            for i, (images, labels, metadata) in enumerate(train_loader):
                labels = labels.cuda()
                images = images.cuda()

                lr = optimizer.param_groups[0]['lr']
                optimizer.zero_grad()
        
                # Autocast to float 16
                with torch.cuda.amp.autocast():
                    output = model(images)
                    labels = labels.half()
                    if metadata["mask"] != "None":
                        face_mask = metadata["mask"].cuda().half()

                    face_weight = configs["face_weight"]

                    if face_weight == "schedule_cresce":
                        face_weight = epoch_percentage*0.8 + 0.2 #new, tava dando erro na loss crescente, talvez pq fosse 0 no início
                    elif face_weight == "schedule_baixa":
                        face_weight = 1 - epoch_percentage
                    elif isinstance(face_weight, str):
                        raise ValueError("Face weight config n faz sentido")

                    full_loss = face_loss = 0
                    
                    if face_weight < 1: 
                        full_loss = metrics["loss"](output, labels)

                    if face_weight > 0:
                        face_labels = labels*face_mask
                        face_output = output*face_mask
                        face_loss = metrics["loss"](face_output, face_labels)  #computing loss only on facemask

                    loss = (1 - face_weight)*full_loss + face_weight*face_loss
                    loss = torch.clip(loss, -5, 5)  # control half instability
                scaler.scale(loss).backward()
                
                if "grad_clip" in configs:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), configs["grad_clip"])

                # If using torchmetrics SAM as loss function, reset its memory after epoch
                # If using torchmetrics loss, need to reset
                if hasattr(metrics["loss"], "reset") or (index == 'torchmetrics_SAM'):
                    metrics['loss'].reset()

                if configs['wandb']:
                    wandb.log({f"Train Loss {metrics['loss'].__class__.__name__}": loss})
                    wandb.log({f"Train face loss": face_loss})
                    wandb.log({f"Train full loss": full_loss})
                    wandb.log({"LR": lr})
                    wandb.log({"face_weight": face_weight})
                    wandb.log({"epoch": epo})

                # optimizer.step in AMP mode, avoids gradient corruption due to low precision
                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                losses.update(loss.data)

                progress_bar.set_postfix(
                    desc=f'[epoch: {epo + 1:d}], iteration: {iteration:d}, loss: {loss.item()}'
                )
                iteration = iteration+1

            ## Epoch end logistics
            epoch_end(val_loader, model, metrics, configs, epo, iteration, optimizer, lr, losses, fixed_checkpoint_name, debug=configs["debug"], cw=cw, ce=ce)
        

# Validate
def validate(val_loader, model, metrics, configs):
    global CHANNEL_WEIGHT
    global CHANNEL_ERRORS #new

    model.eval()
    
    # Initialize an averager for each metric included in criterions
    averages: Dict[str, AverageMeter] = {metric_name: AverageMeter() for metric_name in metrics.keys()}
    averages.update({f"face_{metric_name}": AverageMeter() for metric_name in metrics.keys()})
    cpu_val = configs.get("cpu_val", False)
    if cpu_val:
        model = model.cpu()

    channel_error = torch.tensor(np.zeros((len(val_loader), 61))) #criando um tensor de 0s 
    i=0

    for input, target, metadata in tqdm(val_loader, desc="Validating...", leave=True):
        if not cpu_val:
            input = input.cuda()
            target = target.cuda()
            channel_error = channel_error.cuda() #new 
        face_mask = metadata["mask"]  # [B, C, Y, X], [B, str]
        
        face_eval = False
        if torch.is_tensor(face_mask):
            face_eval = True
            if not cpu_val:
                face_mask = face_mask.cuda()

        with torch.no_grad():
            output = model(input)

            # Computa todas as metricas fornecidas e salva resultado em results
            results = {metric_name: metric(output, target) for metric_name, metric in metrics.items()}
            if face_eval:
                results.update({f"face_{metric_name}": metric(output*face_mask, target*face_mask) for metric_name, metric in metrics.items()})
                
            # Update averagers
            for metric_name, metric in results.items():
                averages[metric_name].update(metric.data)

            #calculating channel weights 
            pixel_diff = output - target
            pixel_diff = torch.abs(pixel_diff)*face_mask
            spectral_diff = torch.mean(pixel_diff,(2,3))

            if len(spectral_diff.size())>1: 
                spectral_diff = spectral_diff[0, :]
            assert len(spectral_diff) == 61

            channel_error[i,:] = spectral_diff

        i = i+1# contador de loops

    CHANNEL_WEIGHT = torch.mean(channel_error,0)
    CHANNEL_ERRORS = torch.mean(channel_error,0) #new
    CHANNEL_WEIGHT = (CHANNEL_WEIGHT - torch.min(CHANNEL_WEIGHT))/(torch.max(CHANNEL_WEIGHT)- torch.min(CHANNEL_WEIGHT))
    c_min = torch.argmin(CHANNEL_WEIGHT)
    CHANNEL_WEIGHT[c_min] = 0.05
    CHANNEL_WEIGHT = CHANNEL_WEIGHT.tolist()
    CHANNEL_ERRORS = CHANNEL_ERRORS.tolist() #new

    assert len(CHANNEL_WEIGHT) == 61
    #print('configs: ', configs['channel_w'])
    configs['channel_w'] = CHANNEL_WEIGHT
    print('CHANNEL WEIGHTS: ',CHANNEL_WEIGHT)

    if "ssim" in metrics:
        if hasattr(metrics["ssim"], "reset"):
            metrics["ssim"].reset() #Cleaning memory cache from torchmetrics loss SSIM

    # Getting loss param
    index = configs.get("loss", None)

    # If using torchmetrics SAM as loss function, reset its memory after epoch
    if(index == 'torchmetrics_SAM'):
        metrics['loss'].reset()

    if cpu_val:
        model = model.cuda()

    # Return averages per metric
    return {metric_name: averages[metric_name].avg for metric_name in averages.keys()}


def epoch_end(val_loader, model, metrics, configs, epo, iteration, optimizer, lr, losses, fixed_checkpoint_name, debug=False, cw=[], ce=[]):
    global RECORD_METRIC

    cw.append(CHANNEL_WEIGHT)#new
    ce.append(CHANNEL_ERRORS)#new

    save_cw(cw,configs['fixed_checkpoint_name'],configs['save_checkpoint_path'], 'cw')#new
    save_cw(ce,configs['fixed_checkpoint_name'],configs['save_checkpoint_path'], 'ce')#new

    if hasattr(metrics["loss"], "epoch_step"):
        metrics["loss"].epoch_step()

    val_metrics = validate(val_loader, model, metrics, configs)
    val_reference = configs.get("val_reference", "loss")
    best_direction = configs.get("best_direction", "min")
    
    val_metric = val_metrics[val_reference]
    if torch.is_tensor(val_metric):
        val_metric = val_metric.item()
    
    if configs['wandb']:
        for metric_name, metric_value in val_metrics.items():
            wandb.log({f'Valid {metric_name}': metric_value})
    
    ckpt_path = configs['save_checkpoint_path']

    print(f"Checking if it is the best weight by {val_reference} {best_direction}, current {val_reference}: {val_metric} VS current best {val_reference}: {RECORD_METRIC}")
    if best_direction == "min":
        if RECORD_METRIC is None:
            RECORD_METRIC = math.inf
        if val_metric < RECORD_METRIC:
            print(f'NEW BEST MODEL! with {val_reference}: {val_metric} Saving to {ckpt_path}')
            RECORD_METRIC = val_metric
            fixed_save_checkpoint(configs['save_checkpoint_path'], epo, iteration, model, optimizer, RECORD_METRIC, fixed_checkpoint_name, disable=debug)
    elif best_direction == "max":
        if RECORD_METRIC is None:
            RECORD_METRIC = -math.inf
        if val_metric > RECORD_METRIC:
            print(f'NEW BEST MODEL! with {val_reference}: {val_metric} Saving to {ckpt_path}')
            RECORD_METRIC = val_metric
            fixed_save_checkpoint(configs['save_checkpoint_path'], epo, iteration, model, optimizer, RECORD_METRIC, fixed_checkpoint_name, disable=debug)
    
    epoch_loss = losses.avg
    if configs['wandb']:
        wandb.log({"train_epoch_loss": epoch_loss})
        wandb.log({"best_val_loss": RECORD_METRIC})

    print(f"Iter {iteration}\nepoch {epo}\nlearning rate {lr},\ntrain loss avg: {epoch_loss}\nval {val_reference}: {val_metric}\nglobal record {val_reference} {RECORD_METRIC}")
    print("\nValidation metrics: ")
    for metric_name, metric_value in val_metrics.items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training hyper skin challenge baseline")

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file", default=None
    )
    parser.add_argument("--debug", action="store_true", help="Force wandb off")

    args = parser.parse_args()

    configs = read_yaml(args.config_file)
    configs["debug"] = args.debug
    if args.debug:
        print("Forcing WANDB off")
        configs["wandb"] = False
        
    f_configurations = {}
    config_flatten(configs, f_configurations)

    print("Flattened configuration:")
    for k, v in f_configurations.items():
        print(f"{k}: {v}")

    # loss function
    criterion = loss_select(configs)
    if hasattr(criterion, "set_max_epochs"):
        criterion.set_max_epochs(configs["train"]["epochs"])

    print(f"Loss: {criterion}")
    
    if criterion is None:
        print("WARNING: Defaulting to original MRAE loss")
        criterion = Loss_MRAE()

    metrics = {"loss": criterion.cuda()}
    cpu_val = configs.get("cpu_val", False)
    remove_ssim_sam = configs.get("remove_ssim_sam", False)
    if remove_ssim_sam:
        print("WARNING: Not performing SSIM and SAM in validation")
    else:
        if cpu_val:
            metrics["ssim"] = StructuralSimilarityIndexMeasure(compute_with_cache=False)
            metrics["sam"] = SAMScore()
            print("WARNING: Performing validation in the CPU")
        else:
            metrics["ssim"] = StructuralSimilarityIndexMeasure(compute_with_cache=False).cuda()  # torchmetrics implementation
            metrics["sam"] = SAMScore().cuda()  # official challenge implementation

    # visnir is the only task
    base_path = configs["base_path"]
    if not os.path.isdir(base_path):
        base_path = os.getenv(base_path)
    assert os.path.isdir(base_path), f"{base_path} is not a directory"

    if configs["task"] == "visnir":
        model = model_select(configs)
        train_transform = transform_factory(configs["train"].get("transform_index", None))
        val_transform = transform_factory(configs["valid"].get("transform_index", None))
        fold = configs.get("fold", None)
        train_dataset = FixedDataset("train", base_path=base_path, transform=train_transform, preprocessing=configs.get("preprocessing", None), fold=fold)
        val_dataset = FixedDataset("val", base_path=base_path, transform=val_transform, preprocessing=configs.get("preprocessing", None), fold=fold)

        f_configurations["readable_train_transform"] = str(train_transform)
        f_configurations["readable_val_transform"] = str(val_transform)
        f_configurations["readable_train_transform"] = train_dataset.init_str()
        f_configurations["readable_val_transform"] = val_dataset.init_str()
    else:
        raise ValueError("The only task is visnir")

    print(f'Train dataset size = {len(train_dataset)}')
    print(f'Valid dataset size = {len(val_dataset)}')

    # If fine-tuning    
    pretrained_model_path = configs['pretrained_path']
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, map_location=torch.device('cpu')) #Avoid infinity loading
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                            strict=True)

    # Put model on GPU, training without cuda is not viable so not even checking
    model.cuda()

    print('Parameters number is ', sum(param.numel() for param in model.parameters()))

    opt = configs.get("opt", None)
    if opt is None:
        print("Using default Adam")
        optimizer = optim.Adam(model.parameters(), lr=configs['train']['lr'], betas=(0.9, 0.999))
    elif opt == "AdamW":
        print("Using AdamW with weight decay 1e-5")
        optimizer = optim.AdamW(model.parameters(), lr=configs['train']['lr'], weight_decay=1e-5)

    iteration = 0
    start_epoch = 0
    best_loss = 1000
    # If continue training from checkpoint
    resume_file = configs['continue_model_path']
    if resume_file is not None:
        if os.path.isfile(resume_file):
            print("=> Resuming training from checkpoint: '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                best_loss = checkpoint['best_loss']
            except:
                print('Forcing best_loss = 1000 due to code version mismatch')
                best_loss = 1000

            print(f"=> Starting train from epoch = {start_epoch}, iteration = {iteration} and best_loss = {best_loss}")

    if not os.path.exists(configs['save_checkpoint_path']):
        os.makedirs(configs['save_checkpoint_path'])

    total_iteration = len(train_dataset)

    scheduling = configs.get("scheduling", None)
    if isinstance(scheduling, float):
        step_size = len(train_dataset)//configs['train']['batch_size']
        print(f"Using StepLR with step {scheduling}, step_size {step_size}")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=scheduling)
    elif scheduling is None:
        print("Using default cosine annealing LR")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iteration, eta_min=1e-6)
    elif scheduling == "disable":
        print("Disable scheduling, fixed slow LR advised")
        scheduler = None

    run=None
    if configs['wandb']:
        config_file = args.config_file
        if config_file is not None:
            name = os.path.basename(config_file).replace(".yaml", '')
        else:
            name = None
        if resume_file is not None:
            run = wandb.init(project="hyperskin-challenge",
                             reinit=True,
                             config=f_configurations,
                             notes="Running experiment",
                             entity="rainbow-ai",
                             id=configs['resume_wandb_id'], resume="must",
                             name=name)
        else:
            run = wandb.init(project="hyperskin-challenge",
                             reinit=True,
                             config=f_configurations,
                             notes="Running experiment",
                             entity="rainbow-ai",
                             name=name)
    
    if configs.get("torchinfo", False):
        import torchinfo
        torchinfo.summary(model, depth=2)
    
    try:
        train(model, metrics, configs, train_dataset, val_dataset, configs['train']['epochs'], total_iteration, start_epoch, iteration)
    except KeyboardInterrupt:
        print("CTRL-C, training interrupted")

    # On training end perform the evaluate script
    if fold is None:
        print("Attempting to evaluate best model... If network doesn't output 1024 resolution, metrics will be only saved locally.")
        model, resume_file, best_loss = load_best_ckpt(model, configs, exact_only=True)

        eval_args = {"post_processing": None,
                     "save": None,
                     "cpu": False,
                     "nworkers": 1}
        torch.cuda.empty_cache()
        val_metrics = evaluation(val_dataset, model, metrics, configs, argparse.Namespace(**eval_args), "val")
    else:
        print("Not evaluating with fold != None, perform manual evaluation.")

    wandb.finish()