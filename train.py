from datasets.dataset import HDTDataset, DatasetProperties
import argparse
from utils.utils_config import get_config
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms
from models.PDT import PDT
from torchsummary import summary

# Create an inception resnet (in eval mode):


def main(args):
    # cfg = get_config(args.config)
    # d_properties = DatasetProperties(path=cfg.path_to_dataset, 
    #                                  same_label_pairs=cfg.load_same_label_pairs, 
    #                                  save_same_label_pairs=cfg.save_same_label_pairs,
    #                                  subindex_for_label=cfg.load_subindex_for_label,
    #                                  save_subindex_for_label=cfg.save_subindex_for_label)
    # dataset = HDTDataset(d_properties)
    img = Image.open('/Users/robinin/face_recognition/datasets/clean/NIR/s2_NIR_00071_001.bmp')
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    ])
    input = transform(img)
    # print(input)
    translator=PDT(pool_features=6,use_se=False, use_bias=False, use_cbam=True)
    summary(translator, (3,112,112))
    print(translator)
    # def initialize_weights(m):
    #     classname = m.__class__.__name__

    #     if (classname.find('Linear') != -1):
    #         m.weight.data.normal_(mean = 0, std = 0.01)
    #     if (classname.find('Conv') != -1):
    #         m.weight.data.normal_(mean = 0.5, std = 0.01)

    # translator.apply(initialize_weights)

    output = translator.forward(input.unsqueeze(0))
    # translator.forward()
    
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    embed = resnet(output)
    print(f"embed: is {embed.size()}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="...")
    parser.add_argument("--config", type=str, default = 'configs/base', help="py config file")
    main(parser.parse_args())