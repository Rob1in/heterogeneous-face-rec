from datasets.dataset import HDTDataset, DatasetProperties, NIRDataset, VISDataset, read_CASIA
from torchvision import transforms
from models.PDT import PDT
import torch
import hydra
from facenet_pytorch import InceptionResnetV1
from omegaconf import DictConfig
from sklearn import metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from PIL import Image

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

    path_to_image = "/Users/robinin/ir_rgb_pairs/1_ir.jpeg"

    input_img  = Image.open(path_to_image)
    img = test_transform(input_img).unsqueeze(0)
    translation = 0
    embedding = 0
    N = 100
    for i in range(N):
        t0 = time.time()
        three_channel=translator(img)
        t1 = time.time()
        translation +=  t1 - t0
        _ = resnet(three_channel)
        embedding+= time.time() -t1
        
    print(f"Time to translate is {translation/N}")
    print(f"Time to embed is {embedding/N}")
if __name__ == "__main__":
    main()