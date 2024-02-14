import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from onnx2torch import convert

onnx_model_path = '/Users/robinin/Downloads/face_recognition_sface_2021dec.onnx'
sface = convert(onnx_model_path)
class DeepContrastiveLoss(torch.nn.Module):

    def __init__(self,
                 framework:str='facenet_pytorch', 
                 pretrained_on:str='vggface2',
                 margin=2.0):
        super(DeepContrastiveLoss, self).__init__()
        self.margin = margin
        if framework == 'facenet_pytorch':
            self.face_recognizer = InceptionResnetV1(pretrained=pretrained_on).eval()
        if framework == 'sface':
            self.face_recognizer = convert(onnx_model_path).eval()
        self.face_recognizer.requires_grad_(False)
    def forward(self, output1, output2, same_label:bool):
        embedding_1 = self.face_recognizer(output1)
        embedding_2 = self.face_recognizer(output2)
        label = same_label.int()
        euclidean_distance = F.pairwise_distance(embedding_1, embedding_2)
        pos = label * torch.pow(euclidean_distance, 2)
        neg = (1- label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive