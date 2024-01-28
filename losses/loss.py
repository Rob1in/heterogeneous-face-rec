import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

class DeepContrastiveLoss(torch.nn.Module):

    def __init__(self, pretrained_on:str='vggface2',  margin=2.0):
        super(DeepContrastiveLoss, self).__init__()
        self.margin = margin
        self.resnet = InceptionResnetV1(pretrained=pretrained_on).eval()
        self.resnet.requires_grad_(False)
    def forward(self, output1, output2, same_label:bool):
        embedding_1 = self.resnet(output1)
        embedding_2 = self.resnet(output2)
        label = same_label.int()
        euclidean_distance = F.pairwise_distance(embedding_1, embedding_2)
        pos = label * torch.pow(euclidean_distance, 2)
        neg = (1- label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive