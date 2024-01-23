from torch.utils.data import Dataset
import torch
import os
from tqdm import tqdm
import json

def random_element_in_list(l):
    index = int(torch.rand(1).item()*len(l))
    return l[index]

def read_CASIA(file_name:str):
    base_name, _ = os.path.splitext(file_name)
    return tuple(base_name.split('_'))
    
class DatasetProperties():
    def __init__(self, 
                 path:str, 
                 nir_folder:str = 'NIR', 
                 vis_folder:str = 'VIS', 
                 same_label_pairs: str = None, 
                 save_same_label_pairs: str = None, 
                 subindex_for_label: str = None, 
                 save_subindex_for_label:str = None):
        self.path = path
        self.nir_folder = os.path.join(path, nir_folder)
        self.vis_folder = os.path.join(path, vis_folder)
        self.nir_images =self.get_image_names_with_modality('NIR')
        self.vis_images = self.get_image_names_with_modality('VIS')
        if same_label_pairs is not None:
            with open(same_label_pairs, 'r') as json_file:
                self.same_label_pair = json.load(json_file)
        else:
            self.same_label_pair = {}
            for img_name in tqdm(self.nir_images, desc ="Creating genuine pairs:"):
            # for img_name in self.nir_images:
                modality, label, sub_index = read_CASIA(img_name)
                key = f"{modality}_{label}_{sub_index}"
                # constructing so it's like [MOD_LABEL_SUBINDEX][VIS][NIR]
                self.same_label_pair[key] = [[],[]] #TODO: CHECK IF IT ALREADY EXISTS
                for candidate in self.nir_images:
                    candidate_modality, candidate_label, candidate_sub_index = read_CASIA(candidate)
                    if candidate_label == label and sub_index != candidate_sub_index:
                        self.same_label_pair[key][0].append(f"{candidate_modality}_{candidate_label}_{candidate_sub_index}")
                        
                for candidate in self.vis_images:
                    candidate_modality, candidate_label, candidate_sub_index = read_CASIA(candidate)
                    if candidate_label == label:
                        self.same_label_pair[key][1].append(f"{candidate_modality}_{candidate_label}_{candidate_sub_index}")
            
            if save_same_label_pairs is not None:
                with open(save_same_label_pairs, 'w') as json_file:
                    json.dump(self.same_label_pair, json_file)
                
        if subindex_for_label is not None:
            with open(subindex_for_label, 'r') as json_file:
                self.all_sub_index_list = json.load(json_file)
        else:
            self.all_sub_index_list = {}
            for img_name in tqdm(self.nir_images, desc ="Creating dictionnary of sub_index list:"):
                modality, label, sub_index = read_CASIA(img_name) #here modality should be 1
                key = f"NIR_{label}"
                if key not in self.all_sub_index_list:
                    self.all_sub_index_list[key] = [f"{key}_{sub_index}"]
                else:
                    self.all_sub_index_list[key].append(f"{key}_{sub_index}")
                    
            for img_name in tqdm(self.vis_images, desc = "Creating dictionnary of sub_index list:"):
                modality, label, sub_index = read_CASIA(img_name) #here modality should be 1
                key = f"VIS_{label}"
                if key not in self.all_sub_index_list:
                    self.all_sub_index_list[key] = [f"{key}_{sub_index}"]
                else:
                    self.all_sub_index_list[key].append(f"{key}_{sub_index}")                
            
            if save_subindex_for_label is not None:                
                with open(save_subindex_for_label, 'w') as json_file:
                    json.dump(self.all_sub_index_list, json_file)
                

    def get_image_names_with_modality(self, modality:str):
        """_summary_
        Filter files in tree that don't match start with modality name. 
        Also removes extension.
        Args:
            modality (str): VIS or NIR

        Returns:
            [str]: List of valid file names
        """        
        if modality == 'NIR':
            files = os.listdir(self.nir_folder)
            res = [os.path.splitext(file)[0]  for file in files if file.startswith('NIR')]
        
        if modality == 'VIS':
            files = os.listdir(self.vis_folder)
            res = [os.path.splitext(file)[0] for file in files if file.startswith('VIS')]
        return res
    
class HDTDataset(Dataset):
    """_summary_
    Dataset for Heterogeneous Domain Transformer training
    Args:
        dataset (DatasetProperties): _description_
        probability_to_same_class (float): probability to draw a pair of image of the same class/person
        probability_to_get_only_nir (float): probability to draw a pair of NIR/NIR images
    """    
    def __init__(self, dataset_properties: DatasetProperties,
                 probability_to_same_class:float =.5,
                 probability_to_get_only_nir: float =.5,
                 transform=None) -> None:
        self.dp = dataset_properties
        self.probability_to_same_class = probability_to_same_class
        self.probability_to_only_nir = probability_to_get_only_nir
        self.transform = transform
        
    def __len__(self):
        return len(self.nir_images)
           
    def __getitem__(self, index):
        """

        Args:
            index (_type_):

        Returns:
            _type_: _description_
        """        
        # img_folder = self.nir_folder
        img_name = self.dp.nir_images[index]
        modality, label, sub_index = read_CASIA(img_name)
        
        same_class = torch.rand(1).item() < self.probability_to_same_class
        only_nir = torch.rand(1).item() < self.probability_to_only_nir
        
        print(f"same class: {same_class}, only_nir: {only_nir}")
        if same_class and only_nir:    
            img2_name = random_element_in_list(self.dp.same_label_pair[img_name][0])
        
        if same_class and not only_nir:
            img2_name = random_element_in_list(self.dp.same_label_pair[img_name][1])
        
        if not same_class:
            
            #FIND A NEW LABEL
            img2_name = random_element_in_list(self.dp.vis_images)
            _, label2, _ = read_CASIA(img_name)
            while (label2 == label):
                img2_name = random_element_in_list(self.dp.nir_images)
                _, label2, _ = read_CASIA(img2_name)
            
            if only_nir:
                img2_name = random_element_in_list(self.dp.all_sub_index_list[f'NIR_{label2}'])
            else:
                img2_name = random_element_in_list(self.dp.all_sub_index_list[f'VIS_{label2}'])
             
        return img_name, img2_name
    
    def get_image_names_with_modality(self, modality:str):
        """_summary_
        Filter files in tree that don't match start with modality name. 
        Also removes extension.
        Args:
            modality (str): VIS or NIR

        Returns:
            [str]: List of valid file names
        """        
        if modality == 'NIR':
            files = os.listdir(self.nir_folder)
            res = [os.path.splitext(file)[0]  for file in files if file.startswith('NIR')]
        
        if modality == 'VIS':
            files = os.listdir(self.vis_folder)
            res = [os.path.splitext(file)[0] for file in files if file.startswith('VIS')]
        return res