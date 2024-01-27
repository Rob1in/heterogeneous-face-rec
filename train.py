from datasets.dataset import HDTDataset, DatasetProperties
import argparse
from utils.utils_config import get_config


from datasets.dataset import HDTDataset, DatasetProperties
import argparse
from utils.utils_config import get_config
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
from models.PDT import PDT
from torchsummary import summary
import torch
from torch.utils.data import DataLoader
from losses.loss import DeepContrastiveLoss
from torchvision import transforms

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE IS {device}")
    cfg = get_config(args.config)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   ),
        
    ])

    train_transform = transforms.Compose([
        test_transform,
        transforms.RandomHorizontalFlip(0.5)
    ])

    #Dataset
    d_properties = DatasetProperties(path=cfg.path_to_dataset, 
                                     same_label_pairs=cfg.load_same_label_pairs, 
                                     save_same_label_pairs=cfg.save_same_label_pairs,
                                     subindex_for_label=cfg.load_subindex_for_label,
                                     save_subindex_for_label=cfg.save_subindex_for_label)
    train_dataset = HDTDataset(d_properties, custom_transform=train_transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size, 
                                  shuffle=True, 
                                  num_workers=cfg.dl_num_workers)
    #TODO: train val test split
    
    #Model
    translator=PDT(pool_features=6,use_se=False, use_bias=False, use_cbam=True)
    translator.to(device)
    #TODO: Make it configurable
    

    #Optimizer and Loss
    optimizer = torch.optim.Adam(translator.parameters(), lr=cfg.learning_rate)
    criterion = DeepContrastiveLoss(pretrained_on = cfg.pretrained_on, 
                                    margin=cfg.loss_margin)

    
    criterion.to(device)
    train_loss = []

    
    if cfg.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profiler_logs_path),
            record_shapes=True,
            with_stack=True)
        
        prof.start()
    translator.train()
    
    for epoch in range(cfg.n_epoch):
        train_epoch_loss = 0
        
        # for i, (img1, img2, same_label, only_nir) in enumerate(train_dataloader):
        #     print(f"size of img1 is {img1.size()}")
        #     print(f"size of only_nnir is {only_nir.size()}")
        #     output_1 = translator(img1)
        #     if only_nir:
        #         output_2 = translator(img2)
        #     else:
        #         output_2 = img2
        #     loss = criterion(output_1, output_2, same_label[i].item())
        #     train_epoch_loss += loss.item()
        #     loss.backward()
        #     optimizer.step()
        for batch_index, (img1, img2, same_label, only_nir) in enumerate(train_dataloader):
            if cfg.profile:
                prof.step()
                if batch_index >5:
                    break
            m = only_nir.size()[0]
            img1, img2, same_label, only_nir = map(lambda x: x.to(device), [img1, img2, same_label, only_nir])
            for i in range(m):
                output_1 = translator(img1[i].unsqueeze(0))
                if only_nir[i]:
                    output_2 = translator(img2[i].unsqueeze(0))
                else:
                    output_2 = img2[i].unsqueeze(0)
                loss = criterion(output_1, output_2, same_label[i].item())
                train_epoch_loss += loss.item()
                loss.backward()
            optimizer.step()
        train_epoch_loss /= len(train_dataset)
        train_loss.append(train_epoch_loss)
        
        print("Epoch [{}/{}] ----> Training loss :{} \n".format(epoch+1,cfg.n_epoch,train_epoch_loss))    
    
    if cfg.profile:
        prof.stop()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="...")
    parser.add_argument("--config", type=str, default = 'configs/base', help="py config file")
    main(parser.parse_args())