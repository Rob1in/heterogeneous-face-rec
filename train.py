from datasets.dataset import HDTDataset, DatasetProperties
import argparse
from utils.utils_config import get_config
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms


# Create an inception resnet (in eval mode):


def main(args):
    # cfg = get_config(args.config)
    # d_properties = DatasetProperties(path=cfg.path_to_dataset, 
    #                                  same_label_pairs=cfg.load_same_label_pairs, 
    #                                  save_same_label_pairs=cfg.save_same_label_pairs,
    #                                  subindex_for_label=cfg.load_subindex_for_label,
    #                                  save_subindex_for_label=cfg.save_subindex_for_label)
    # dataset = HDTDataset(d_properties)

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    img = Image.open('/Users/robinin/face_recognition/datasets/test/000002/s1_VIS_00002_006.jpg')
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    # Step 3: Apply the transformation to the image
    tensor_image = transform(img)
    print(f"tensor shape is {tensor_image}")
    res = resnet(tensor_image.unsqueeze(0))
    print(res)
    # for i in range(10):
    #     print(dataset.__getitem__(i))
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="...")
    parser.add_argument("--config", type=str, default = 'configs/base', help="py config file")
    main(parser.parse_args())