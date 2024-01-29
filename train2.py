from datasets.dataset import HDTDataset, DatasetProperties
import argparse
from torchvision import transforms
from models.PDT import PDT
import torch
from torch.utils.data import DataLoader
from losses.loss import DeepContrastiveLoss
import hydra

from omegaconf import DictConfig
from utils.save_model import save_checkpoint, create_checkpoint_folder, get_summary_writer
import os

@hydra.main(version_base=None, config_path="./configs", config_name="base")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE FOUND: {device}")
    
   
    checkpoints_folder_path = create_checkpoint_folder()
    writer = get_summary_writer()
    
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
    d_properties = DatasetProperties(path=cfg.dataset.path, 
                                     nir_folder= cfg.dataset.nir_folder, 
                                     vis_folder= cfg.dataset.vis_folder,
                                     same_label_pairs=cfg.dataset.load_same_label_pairs, 
                                     save_same_label_pairs=cfg.dataset.save_same_label_pairs,
                                     subindex_for_label=cfg.dataset.load_subindex_for_label,
                                     save_subindex_for_label=cfg.dataset.save_subindex_for_label)
    
    train_dataset = HDTDataset(d_properties, custom_transform=train_transform)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.dataloader.batch_size, 
                                  shuffle=True, 
                                  num_workers=cfg.dataloader.num_workers)
    #TODO: train val test split
    
    #Model
    translator=PDT(pool_features=cfg.PDT.pool_features, use_se=False, use_bias=False, use_cbam=True)
    translator.to(device)
    #TODO: Make it configurable
    

    #Optimizer and Loss
    optimizer = torch.optim.Adam(translator.parameters(), lr=cfg.learning_rate)
    criterion = DeepContrastiveLoss(pretrained_on = cfg.loss.pretrained_on,
                                    margin=cfg.loss.margin)

    
    criterion.to(device)
    train_loss = []
    translator.train()
        
    for epoch in range(cfg.n_epoch):
        train_epoch_loss = 0
        for batch_index, (img1, img2, same_label, only_nir) in enumerate(train_dataloader):
            img1, img2, same_label, only_nir = map(lambda x: x.to(device), [img1, img2, same_label, only_nir])

            # Translate all images in the batch in one go
            output_1 = translator(img1)
            output_2 = translator(img2)
            
            # Use only_nir to select the translated image or the original image
            output_2[~only_nir] = img2[~only_nir]

            # Calculate loss using vectorized operations
            loss = criterion(output_1, output_2, same_label)
            train_epoch_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break
        
        train_epoch_loss /= len(train_dataloader.dataset)
        writer.add_scalar('training loss', train_epoch_loss, epoch*len(train_dataloader))
        train_loss.append(train_epoch_loss)
        print("Epoch [{}/{}] ----> Training loss :{} \n".format(epoch+1,cfg.n_epoch,train_epoch_loss))
        
        is_best, best_acc = True, .3
        if epoch %cfg.save_checkpoint:
            
            save_checkpoint({
                'epoch': epoch + 1,
                'optimizer' : optimizer.state_dict(),
                'best_acc': best_acc,
                'model_state_dict': translator.state_dict(), 
                },
                is_best,
                it = epoch*len(train_dataloader),
                path = os.path.join(checkpoints_folder_path))
            
        if cfg.validate:
            translator.eval()
            val_loss = 0
            for batch_index, (img1, img2, same_label, only_nir) in enumerate(train_dataloader):
                img1, img2, same_label, only_nir = map(lambda x: x.to(device), [img1, img2, same_label, only_nir])
                output_1 = translator(img1)
                output_2 = translator(img2)
                output_2[~only_nir] = img2[~only_nir]
                loss = criterion(output_1, output_2, same_label)
                val_loss += loss.item()
                break
            writer.add_scalar('val loss', val_loss, epoch*len(train_dataloader))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="...")
    parser.add_argument("--config", type=str, default = 'configs/base', help="py config file")
    main()