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
    translator.train()
    
    # for epoch in range(cfg.n_epoch):
    #     train_epoch_loss = 0
    #     for batch_index, (img1, img2, same_label, only_nir) in enumerate(train_dataloader):
    #         m = only_nir.size()[0]
    #         img1, img2, same_label, only_nir = map(lambda x: x.to(device), [img1, img2, same_label, only_nir])
    #         for i in range(m):
    #             output_1 = translator(img1[i].unsqueeze(0))
    #             if only_nir[i]:
    #                 output_2 = translator(img2[i].unsqueeze(0))
    #             else:
    #                 output_2 = img2[i].unsqueeze(0)
    #             loss = criterion(output_1, output_2, same_label[i].item())
    #             train_epoch_loss += loss.item()
    #             loss.backward()
    #         optimizer.step()
    #     train_epoch_loss /= len(train_dataset)
    #     train_loss.append(train_epoch_loss)
        
    #     
        
        
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_epoch_loss /= len(train_dataloader.dataset)
        train_loss.append(train_epoch_loss)
        print("Epoch [{}/{}] ----> Training loss :{} \n".format(epoch+1,cfg.n_epoch,train_epoch_loss))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="...")
    parser.add_argument("--config", type=str, default = 'configs/base', help="py config file")
    main(parser.parse_args())