import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class DeepContrastiveLoss(torch.nn.Module):

    def __init__(self, pretrained_on:str='vggface2', device,  margin=2.0):
        super(DeepContrastiveLoss, self).__init__()
        self.margin = margin
        self.resnet = InceptionResnetV1(pretrained=pretrained_on).eval()
        self.resnet.to(device)
        self.resnet.requires_grad_(False)
    def forward(self, output1, output2, same_label:bool):
        """_summary_

        Args:
            output1 (_type_): 
            output2 (_type_): _description_
            same_label (bool): _description_
            only_nir (bool): _description_

        Returns:
            _type_: _description_
        """
        embedding_1 = self.resnet(output1)
        embedding_2 = self.resnet(output2)
        label = int(same_label)
        euclidean_distance = F.pairwise_distance(embedding_1, embedding_2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive