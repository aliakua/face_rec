from torchvision import datasets, models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone_Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.backbone = create_feature_extractor(self.model, ['flatten'])

    def forward(self, x):
        x = self.backbone(x) #['flatten'] for ang - off
        return x
    

class Baseline_Net(nn.Module):
    def __init__(self, backbone, emb_size,  out_num):
        super().__init__()
        self.backbone = backbone
        self.embed = nn.Sequential(
                          nn.Linear(512 , 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(inplace = True),
                          nn.Dropout(0.3),
                          nn.Linear(512, emb_size))
        self.classifier = nn.Linear(emb_size , out_num) # Заменяем Fully-Connected слой на наш линейный классификатор

    def forward(self, x, embed = False):
        x = self.backbone(x)['flatten']
        x = self.embed(x)
        if embed:
            return x
        x = self.classifier(x)
        return x
    
class Angular_clf(nn.Module):
    def __init__(self, emb_size,  out_num):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_size, out_num))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x, embed = False):
        #x = self.backbone(x)
        #x = self.fc(x)
        if embed:
            return x
        # Шаг 1: нормализауем  x и W
        x_norm = F.normalize(x) # 32*128
        W_norm = F.normalize(self.W, dim=0) # 128*500
        # Шаг 2: получим косинусы, так называемые центроиды
        return x_norm @ W_norm

class Angular_Net(nn.Module):
    def __init__(self, backbone, classifier, emb_size):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, emb_size)
        self.classifier = classifier

    def forward(self, x, embed = False):
        x = self.backbone(x)['flatten']
        x = self.fc(x)
        if embed:
            return x
        x = self.classifier(x)
        return x