
import os 
import torch

def save_train_test_split(train_set, test_set, path):
    train_set_path = os.path.join(path, 'train.txt')
    test_set_path = os.path.join(path, 'test.txt')
    
    with open(train_set_path, 'w') as f:
        for label in train_set:
            f.write(label + '\n')
            
    with open(test_set_path, 'w') as f:
        for label in test_set:
            f.write(label + '\n')
            
def load_splits(path):
    train_set = set()
    test_set = set()
    train_set_path = os.path.join(path, 'train.txt')
    test_set_path = os.path.join(path, 'test.txt')
    
    
    with open(train_set_path, 'r') as f:
        for line in f:
            train_set.add(line.strip())
            
            
    with open(test_set_path, 'r') as f:
        for line in f:
            test_set.add(line.strip())
            
    return train_set, test_set
                    
def random_element_in_list(l):
    index = int(torch.rand(1).item()*len(l))
    return l[index]

def read_CASIA(file_name:str):
    base_name, _ = os.path.splitext(file_name)
    return tuple(base_name.split('_'))

def are_same_label(img_name1:str, img_name2: str):
    _, label_1, _ = read_CASIA(img_name1)
    _, label_2, _ = read_CASIA(img_name2)
    return label_1 == label_2

def label_in_set(img_name, split:set):
    _, label, _ = read_CASIA(img_name)
    return label in split    
