from datasets.dataset import HDTDataset, DatasetProperties, NIRDataset, VISDataset, read_CASIA
import argparse
from torchvision import transforms
from models.PDT import PDT
import torch
from torch.utils.data import DataLoader
from losses.loss import DeepContrastiveLoss
import hydra
from facenet_pytorch import InceptionResnetV1
from omegaconf import DictConfig
from utils.save_model import save_checkpoint, create_checkpoint_folder, get_summary_writer
import os
from sklearn import metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

@hydra.main(version_base=None, config_path="./configs", config_name="test")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE FOUND: {device}")
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   ),
        
    ])

    #Dataset
    d_properties = DatasetProperties(path=cfg.dataset.path, 
                                     nir_folder= cfg.dataset.nir_folder, 
                                     vis_folder= cfg.dataset.vis_folder,
                                     same_label_pairs=cfg.dataset.load_same_label_pairs, 
                                     save_same_label_pairs=cfg.dataset.save_same_label_pairs,
                                     subindex_for_label=cfg.dataset.load_subindex_for_label,
                                     save_subindex_for_label=cfg.dataset.save_subindex_for_label)
    test_nir_data = NIRDataset(d_properties, test_transform)
    test_vis_data = VISDataset(d_properties, test_transform)
    

    #Model
    path_to_checkpoint = '/Users/robinin/face_recognition/results/model:vggface2_lr:0.001_bs:64_same-label:0.5_only-nir:0.5_01-30_16-10/checkpoints/checkpoint_12960.pt'
    checkpoint = torch.load(path_to_checkpoint, map_location=torch.device(device))
    translator=PDT(pool_features=cfg.PDT.pool_features, use_se=False, use_bias=False, use_cbam=True)
    translator.load_state_dict(checkpoint['model_state_dict'])
    translator.eval()
    translator.requires_grad_(False)
    translator.to(device)
    resnet = InceptionResnetV1(pretrained=cfg.loss.pretrained_on).eval()
    resnet.requires_grad_(False)
    embeddings = {}
    baseline_embeddings = {}
    baseline = True
    fast = False
    
    for img, img_name in tqdm(test_nir_data, desc="Translating and embedding NIR images"):
        _, label, _ = read_CASIA(img_name)
        if fast:
            if int(label)>20:
                continue
        if baseline:
            untouched_img = img.unsqueeze(0)
            embed = resnet(untouched_img)
            baseline_embeddings[img_name] = embed
            
        translated_img = translator(img.unsqueeze(0)) 
        embed = resnet(translated_img)
        embeddings[img_name] = embed
        
    for img, img_name in tqdm(test_vis_data, desc="Embedding VIS images"):
        _, label, _ = read_CASIA(img_name)
        if fast:
            if int(label)>20:
                continue
        embed = resnet(img.unsqueeze(0))
        embeddings[img_name] = embed
        baseline_embeddings[img_name] = embed
    
    distance = F.pairwise_distance
    
    
    Y_truth = []
    ##MODEL
    Y_pred =[]
    distances = {}
    for img_name1 in tqdm(embeddings, desc = 'Calculating distances'):
        for img_name2 in embeddings:
            if img_name1 == img_name2:
                continue
            if (img_name2, img_name1) in distances:
                continue
            distances[(img_name1, img_name2)] = distance(embeddings[img_name1], embeddings[img_name2])
    for pair in distances:
        modality_1, label_1, sub_index_1 = read_CASIA(pair[0])
        modality_2, label_2, sub_index_2 = read_CASIA(pair[1])
        Y_pred.append(distances[pair].item())
        if label_1 == label_2:
            Y_truth.append(0)
        else:
            Y_truth.append(1)
    
    fpr, tpr, thresholds = metrics.roc_curve(Y_truth, Y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr,roc_auc=roc_auc)
    display.plot()
    
    ##BASELINE
    Y_baseline_pred = []
    baseline_distances = {}        
    for img_name1 in tqdm(baseline_embeddings, desc = 'Calculating baseline distances'):
        for img_name2 in baseline_embeddings:
            if img_name1 == img_name2:
                continue
            if (img_name2, img_name1) in baseline_distances:
                continue
            baseline_distances[(img_name1, img_name2)] = distance(baseline_embeddings[img_name1], baseline_embeddings[img_name2])
    for pair in baseline_distances:
        modality_1, label_1, sub_index_1 = read_CASIA(pair[0])
        modality_2, label_2, sub_index_2 = read_CASIA(pair[1])
        Y_baseline_pred.append(baseline_distances[pair].item())

    fpr, tpr, thresholds = metrics.roc_curve(Y_truth, Y_baseline_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    display_baseline = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr,roc_auc=roc_auc)
    display_baseline.plot()
    
    plt.show()

if __name__ == "__main__":
    main()