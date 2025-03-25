import sys
from tqdm import tqdm
import os
import uuid
import torch
import torch.nn as nn
import argparse
from typing import Dict
from torch.utils.data import DataLoader
import wandb
from typing import Dict, Any, List, Optional
import json
from utils import model_select, load_best_ckpt, read_yaml, plot_cube, config_flatten
from baseline.train_code.utils import AverageMeter
from fixed_dataset import FixedDataset
from transforms import transform_factory
from torchmetrics.image import StructuralSimilarityIndexMeasure
from metrics.sam import SAMScore
from post_processing import select_post_processing
from multiprocessing import Queue, Process, cpu_count
import threading
import torchinfo
from pre_evaluate import pre_eval_model
from pre_evaluate import pre_eval_pixel
import numpy as np 

def gpu_tensors_to_numpy(tensor_dict):
    numpy_dict = {}
    for key, tensor in tensor_dict.items():
        # Check if the tensor is on GPU
        if tensor.is_cuda:
            tensor = tensor.cpu()  # Move tensor to CPU
        numpy_dict[key] = float(tensor.numpy())  # Convert tensor to NumPy array
    return numpy_dict

def save_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def plotter_fn(q: Queue, save_dir: str):
    '''
    Thread to manage saving matplotlib renders
    '''
    ps: List[Process] = []
    while True:
        pkg = q.get()
        if pkg is None:
            break
        else:
            cube, name = pkg

        name = os.path.join(save_dir, name)

        p = Process(target=plot_cube, args=(cube, name, False), daemon=True)
        p.start()
        ps.append(p)

    print("Plotter signalled to stop, waiting for processes to complete for up to 1 minute.")
    for p in ps:
        p.join(timeout=60)
    print("Plotter finished.")


def evaluation(eval_dataset: FixedDataset, 
               model: nn.Module, 
               metrics: Dict[str, object], 
               configs: Dict[str, Any], 
               args: argparse.Namespace,
               mode: str,
               save_dir: Optional[str] = None):
    model.eval()
    list_of_results = [{}]
    face_sam_list = []
    val_loader = DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False, num_workers=min(configs.get("nworkers"), getattr(args, "nworkers", 1)))
    post_proc = None  # houses object to performing requested post processing
    saver = None  # houses object to performing requested saving function

    # Configuring post processing
    if args.post_processing is not None:
        post_proc = select_post_processing(args.post_processing)

    # Configuring saver
    if save_dir is None:
        save_dir = os.path.join(configs["save_checkpoint_path"], configs["fixed_checkpoint_name"] + "_" + mode + "_output")
    
    os.makedirs(save_dir, exist_ok=True)
    if args.save is not None:
        saver = select_post_processing(args.save)
        saver.save_dir = save_dir

    # Initialize an averager for each metric included in criterions
    averages: Dict[str, AverageMeter] = {metric_name: AverageMeter() for metric_name in metrics.keys()}
    averages.update({f"face_{metric_name}": AverageMeter() for metric_name in metrics.keys()})

    device = torch.device("cpu") if args.cpu else torch.device("cuda:0")
    iter = tqdm(val_loader, desc=f"Evaluating {mode}...", leave=True)

    # Parallel saving matplotlib images
    plotter_queue = Queue(cpu_count())
    plotter = threading.Thread(target=plotter_fn, args=(plotter_queue, save_dir), daemon=True)
    plotter.start()
    warn_1024 = False

    for input, target, metadata in iter:
        with torch.no_grad():
            input = input.to(device)
            model = model.to(device)
            output = model(input).detach()
            # Save memory with transfering out of GPU
            model = model.cpu()
            input = input.cpu()

            # Post processing
            if post_proc is not None:
                output, metadata = post_proc(output, metadata)

            if mode != "test":
                target = target.to(device)

                face_mask = metadata["mask"]
                
                face_eval = False
                if torch.is_tensor(face_mask):
                    face_eval = True
                    face_mask = face_mask.to(device)

                if not warn_1024 and not(target.shape[-1] == target.shape[-2] == 1024):
                    tqdm.write("WARNING: Target must be 1024, 1024 spatial resolution for evaluation, even if input is 256. Use the following arguments to achieve that:\n--post_processing upsample_4 --overhide_base_path ICASP --overhide_transform downsample_input_4 --overhide_preprocessing None\nWhere ICASP is the path to the raw original dataset")
                    if configs["wandb"]:
                        tqdm.write("Disabling wandb logging due to running in targets different than 1024x1024")
                        configs["wandb"] = False
                    warn_1024 = True

                # Computa todas as metricas fornecidas e salva resultado em results
                results = {metric_name: metric(output, target) for metric_name, metric in metrics.items()}
                if face_eval:
                    results.update({f"face_{metric_name}": metric(output*face_mask, target*face_mask) for metric_name, metric in metrics.items()})

                # Torchmetrics memory reset
                for metric in metrics.values():
                    if hasattr(metric, "reset"):
                        metric.reset()        

                # Logar dicionario results
                # Guara lista de results, cada item da lista é uma imagem da validação/teste
                list_of_results.append({"ID": metadata["ID"][0], "metric": gpu_tensors_to_numpy(results)})
                print(metadata['ID'])
                print(results)
                #list_of_results.append({metadata["ID"][0]: gpu_tensors_to_numpy(results)})
                print(list_of_results)
                face_sam_list.append(gpu_tensors_to_numpy(results)['face_sam'])
                print(face_sam_list)

                # Update averagers
                for metric_name, metric in results.items():
                    averages[metric_name].update(metric.data)

            # Save on challenge format
            if saver is not None:
                saver(output, metadata)
            save_path = '/mnt/datassd/icasp/RainbowAI/evaluation_results/'
            save_name = configs.get('model')
            
            save_to_json(list_of_results, save_path  + save_name + '_list-of-results.json')
            np.save( save_path + configs.get('model') + '_face-sam.npy',np.array(face_sam_list), allow_pickle=True )
            

            # Save on summarized format
            plotter_queue.put((output.detach().cpu().numpy(), f"{metadata['ID'][0]}_output"))
            if mode != "test":
                plotter_queue.put((target.detach().cpu().numpy(), f"{metadata['ID'][0]}_target"))

    
    if saver is not None and hasattr(saver, "join"):
        saver.join()

    print("Waiting matplotlib plotter to finish saving image summaries...")
    plotter_queue.put(None)
    plotter.join()
    print("Done")

    if mode != "test":
        print("Collecting metrics...")
        val_metrics = {}
        for metric_name in averages.keys():
            val_metrics[metric_name] = averages[metric_name].avg
            if torch.is_tensor(val_metrics[metric_name]):
                val_metrics[metric_name] = val_metrics[metric_name].item()

        metric_str_repr = [f"{name}: {round(value, 2)}" for name, value in val_metrics.items()]
        metric_str_repr = '_'.join(metric_str_repr)

        # Send dict to cloud
        if configs['wandb']:
            print("Logging to wandb...")
            wandb.define_metric("evaluation_metrics")
            for metric_name in val_metrics.keys():
                wandb.define_metric(metric_name, step_metric="evaluation_metrics")
            for metric_name, value in val_metrics.items():
                wandb.log({f"eval_{mode}_{metric_name}": value,
                            "evaluation_metrics": 0})
        else:
            for metric_name, value in val_metrics.items():
                print(f"{metric_name}: {value}")

        #Saving json files in save_checkpoint_path
        additional_parameters = {}
        additional_parameters[f"{mode}_str_repr"] = metric_str_repr
        additional_parameters[f"{mode}_transform"] = str(val_loader.dataset.transform)
        additional_parameters[f"{mode}_base_path"] = val_loader.dataset.base_path
        additional_parameters[f"{mode}_preprocessing"] = val_loader.dataset.preprocessing
        additional_parameters[f"{mode}_post_processing"] = args.post_processing
        additional_parameters[f"{mode}_save"] = args.save
        additional_parameters[f"{mode}_wandb"] = configs["wandb"]
        
        if configs["wandb"]:
            wandb.config.update(additional_parameters)

        val_metrics.update(additional_parameters)

        # Backup local save
        print("Saving to local .json...")
        results_save_path =  os.path.join(configs['save_checkpoint_path'], f"{configs['fixed_checkpoint_name']}_{mode}_results_{metric_str_repr}_{uuid.uuid1()}.json")
        with open(results_save_path, 'w') as fp:
            json.dump(val_metrics, fp)
    else:
        print("Running in test dataset, nothing to log.")


class FoldEvaluator(nn.Module):
    '''
    results are not good!
    '''
    def __init__(self, configs: List[Dict[str, Any]]):
        super().__init__()
        self.configs = configs

        self.channel_w_per_model: Optional[np.ndarray] = None #aqui é a lista de listas 
        
        self.pixel_w_per_model: Optional[np.ndarray] = None

        # Carrega ensemble em ModuleList
        self.models = nn.ModuleList([model_select(config) for config in self.configs])

        # Numero de modelos divide a media
        self.n_models = len(self.models)

        print(f"FoldEvaluation with {len(self.configs)} configs")
    
    def load_ckpts(self):
        # Carrega checkpoints, primeiro do modelo 0 como referencia depois de 1 pra frente
        for i in range(len(self.configs)):
            self.models[i], _, _ = load_best_ckpt(self.models[i], self.configs[i], exact_only=True)

        return self, None, None
    
    def set_channel_w(self, channel_w_per_model: List[List[float]]):
        self.channel_w_per_model = channel_w_per_model
        print(f"These are the best models per channel: {self.channel_w_per_model.argmin(axis=0)}")
    
    def set_pixel_w(self, pixel_w_per_model: List[List[float]]):
        self.pixel_w_per_model = pixel_w_per_model
        self.channel_w_per_model = None
        print(f"These are the best models per pixel: {self.pixel_w_per_model.argmin(axis=0)}")

    def normalize(self, output): 
        norm_output = (output - torch.min(output))/(torch.max(output)-torch.min(output))
        #torch.unsqueeze(norm_output, 0)
        return norm_output
    
    def forward(self, x):
        acum = []
        in_device = x.device
        x = x.cuda()

        # Itera sobre modelos, coloca na gpu, acumula resultado no acum
        for model in tqdm(self.models, desc="Fold evaluating..."):
            model = model.cuda()
            output = model(x).cpu()
            acum.append(self.normalize(output))
            model = model.cpu()
        x = x.to(in_device)

        acum = torch.stack(acum, dim=0)  # [5, 1, 61, 1024, 1024]
        print('first acum shape: ', acum.size())

        # Retorna resultado dividido pelo numero de modelos
        if self.channel_w_per_model is None:
            if self.pixel_w_per_model is None:
                acum = acum.mean(dim=0)  # [1, 61, 1024, 1024]

        if self.channel_w_per_model != None:
            # Clever ana idea
            idxs = self.channel_w_per_model.argmin(axis=0)  # [61], int melhor modelo
         
            new_acum = torch.zeros((acum.shape[1:]), dtype=acum.dtype, device=acum.device)
            
            # Select best output from acumulated outputs
            for c, idx in enumerate(idxs):
                new_acum[:, c] = acum[idx, :, c]
            
            acum = new_acum

        if self.pixel_w_per_model != None :

            idxs = self.pixel_w_per_model.argmin(axis=0)  # [1024,1024], int melhor modelo
            unique_models = len(torch.unique(idxs)) #unique models 
         
            new_acum = torch.zeros((acum.shape[1:]), dtype=acum.dtype, device=acum.device)

            assert len(torch.unique(idxs)<acum.size()[0])

            for i, mod in enumerate(torch.unique(idxs)): 
                temp_mask = idxs==mod #mascara de valores onde idx
                new_acum[:,:, temp_mask] = acum[mod, : , : , temp_mask]
            
            acum = new_acum

        acum = acum.to(in_device)
        return acum
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation pipeline for HyperSkin challenge")
    parser.add_argument("config_file", type=str, help="Path to YAML configuration file", default=None)
    parser.add_argument("--config_files", type=str, help="Path to .txt List of paths to YAML configuration files for each fold, separated by linebreak", default=None)
    parser.add_argument("--wandb_id", type=str, help="WanDB ID in case its not in config file", default=None)
    parser.add_argument("--post_processing", type=str, help="What to do with network output, before metrics. None does nothing", default=None)
    parser.add_argument("--save", type=str, help="How to save predictions. None doesn't save.", default=None)
    parser.add_argument("--overhide_base_path", type=str, help="Use custom base_path instead of config sourced one.", default=None)    
    parser.add_argument("--overhide_transform", type=str, help="Use custom transform instead of config sourced one.", default=None)    
    parser.add_argument("--overhide_preprocessing", type=str, help="Use custom preprocessing instead of config sourced one.", default=None) 
    parser.add_argument("--test", action="store_true", help="Uses test images without targets.")
    parser.add_argument("--channel_select", action="store_true", help="Selects best channels from each ensemble model.")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb.")
    parser.add_argument("--nworkers", type=int, help="Nworkers for data loading", default=1)
    parser.add_argument("--pixel_select", action="store_true", help="Selects best channels from each ensemble model.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for everything.")
    args = parser.parse_args()

    if args.config_file == "folds":
        print("No main config file given, attempting to initialize folds. Fold 0 is used as reference for hyperparameters. Folds:")
        fold_evaluation = True
        with open(args.config_files, 'r') as config_files_txt:
            config_files = [line.strip() for line in config_files_txt.readlines()]
        print(config_files)

        config_list = [read_yaml(config_file) for config_file in config_files]
        configs = config_list[0]
    else:
        fold_evaluation = False
        configs = read_yaml(args.config_file)
    
    if args.no_wandb:
        configs["wandb"] = False

    f_configurations = {}
    config_flatten(configs, f_configurations)

    metrics = {}

    metrics["ssim"] = StructuralSimilarityIndexMeasure(compute_with_cache=False)
    metrics["sam"] = SAMScore()
    if not args.cpu:
        metrics["ssim"] = metrics["ssim"].cuda()
        metrics["sam"] = metrics["sam"].cuda()
    
    if configs["task"] == "visnir":
        if fold_evaluation:
            print("Initializing FoldEvaluator")
            model = FoldEvaluator(config_list)
        else:
            print("Initializing model definition")
            model = model_select(configs)

        base_path = configs["base_path"]
        if not os.path.isdir(base_path):
            base_path = os.getenv(base_path)

        transform_idx = configs["valid"].get("transform_index", None)
        preprocessing = configs.get("preprocessing", None)

        if args.overhide_base_path is not None:
            print(f"Replacing config base_path {base_path} with {args.overhide_base_path}")
            base_path = args.overhide_base_path
            if not os.path.isdir(base_path):
                base_path = os.getenv(base_path)
        
        if args.overhide_transform is not None:
            print(f"Replacing config transform {transform_idx} with {args.overhide_transform}")
            transform_idx = args.overhide_transform

        if args.overhide_preprocessing is not None:
            print(f"Replacing config preprocessing {preprocessing} with {args.overhide_preprocessing}")
            preprocessing = args.overhide_preprocessing

        eval_transform = transform_factory(transform_idx)
        mode = "test" if args.test else "val"

        eval_dataset = FixedDataset(mode,
                                    base_path=base_path, 
                                    transform=eval_transform, 
                                    preprocessing=preprocessing,
                                    fold=None)  # never use fold in evaluation
    else:
        raise ValueError("The only task is visnir")

    print(f'Evaluation {mode} dataset size = {len(eval_dataset)}')

    if fold_evaluation:
        print("Load best CKPTs for all folds")
        model, resume_file, best_loss = model.load_ckpts()
    else:
        print("Loading best CKPT")
        model, resume_file, best_loss = load_best_ckpt(model, configs, exact_only=True)

    torchinfo.summary(model)

    run = None
    save_dir = None
    if configs['wandb']:
        id = configs['resume_wandb_id']
        if id is None:
            print(f"config.yaml resume ID is None, trying argument {args.wandb_id}")
            id = args.wandb_id
            if id is None:
                print("ERROR: This will probably fail given that there is no wandb ID from anywhere.")

        if resume_file is not None:
            run = wandb.init(project="hyperskin-challenge",
                             reinit=True,
                             config=f_configurations,
                             notes="Running experiment",
                             entity="rainbow-ai",
                             id=id, resume="must")
        else:
            print("Not continuing wandb log, you are probably doing fold evaluation")
            run = wandb.init(project="hyperskin-challenge",
                             reinit=True,
                             config=f_configurations,
                             notes=' '.join(config_files),
                             entity="rainbow-ai")
            save_dir = os.path.join("fold_eval_output", ' '.join([os.path.basename(config_file).replace(".yaml", '') for config_file in config_files]).lower().strip())
    
    # Caso channel select folds roda pre_evaluation pra descobrir o channel weight de cada rede
    if args.channel_select:
        print("Pre computing validation weights per model")
        all_cws = np.zeros((len(config_list), 61))
        for i, (each_model, config_k_fold) in enumerate(zip(model.models, config_list)): 
            print(i)
            cw = pre_eval_model(each_model, base_path, config_k_fold, args, eval_transform, preprocessing, use_mask = False)
            all_cws[i,:] = cw
           
        print(all_cws)

        model.set_channel_w(all_cws)
    
    if args.pixel_select:
        print("Pre computing validation weights per model")
        all_pws = torch.tensor(np.zeros((len(config_list), 1024, 1024)))
        for i, (each_model, config_k_fold) in enumerate(zip(model.models, config_list)): 
            print(i)
            pw = pre_eval_pixel(each_model, base_path, config_k_fold, args, eval_transform, preprocessing, use_mask = False)
            all_pws[i,:, :] = pw
           
        print(all_pws)

        model.set_pixel_w(all_pws)

    evaluation(eval_dataset, model, metrics, configs, args, mode, save_dir=save_dir)

    wandb.finish()