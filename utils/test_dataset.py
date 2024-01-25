from datasets.dataset import HDTDataset, DatasetProperties
import argparse
from utils.utils_config import get_config


def main(args):
    cfg = get_config(args.config)
    d_properties = DatasetProperties(path=cfg.path_to_dataset, 
                                     same_label_pairs=cfg.load_same_label_pairs, 
                                     save_same_label_pairs=cfg.save_same_label_pairs,
                                     subindex_for_label=cfg.load_subindex_for_label,
                                     save_subindex_for_label=cfg.save_subindex_for_label)
    dataset = HDTDataset(d_properties)

    for i in range(10):
        print(dataset.__getitem__(i))
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="...")
    parser.add_argument("--config", type=str, default = 'configs/base', help="py config file")
    main(parser.parse_args())