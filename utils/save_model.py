import torch 
import shutil
import hydra
import os
from torch.utils.tensorboard import SummaryWriter


def create_checkpoint_folder():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    checkpoints_folder_path = os.path.join(log_dir, 'checkpoints')

# Create the "checkpoints" folder if it doesn't exist
    if not os.path.exists(checkpoints_folder_path):
        os.makedirs(checkpoints_folder_path)
    else:
        raise FileExistsError(f"'checkpoints' folder already exists at: {checkpoints_folder_path}. If you want to resume training: it's not supported.")
    
    return checkpoints_folder_path
    
    
def save_checkpoint(state, is_best, path, it):
    checkpoint_path = os.path.join(path,f'checkpoint_{it}.pt')
    torch.save(state, checkpoint_path)
    if is_best:
        old_best_models = [f for f in os.listdir(path) if f.startswith('model_best_') and f.endswith('.pt')]
        for old_best_model in old_best_models:
            os.remove(os.path.join(path, old_best_model))
        shutil.copyfile(checkpoint_path, os.path.join(path,f'model_best_{it}.pt'))


def get_summary_writer(create_folder=True):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']
    summaries_folder_path = os.path.join(log_dir, 'summaries')

# Create the "checkpoints" folder if it doesn't exist
    if not os.path.exists(summaries_folder_path):
        os.makedirs(summaries_folder_path)
    else:
        raise FileExistsError(f"'summaries' folder already exists at: {summaries_folder_path}. What are you trying to do?")
    
    return SummaryWriter(summaries_folder_path)