import torch.nn.functional as F
import torch
import torch.nn as nn

class Angular_Loss(nn.Module):
    def __init__(self, loss_type = 'arcface', out_num = 500, s = None, m = None):
        super(Angular_Loss, self).__init__()
        loss_type = loss_type.lower()
        self.out_num = out_num
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 30.0 if not s else s
            self.m = 0.6 if not m else m
        if loss_type == 'sphereface':
            self.s = 30.0 if not s else s
            self.m = 0.35 if not m else m
        if loss_type == 'cosface':          # s and m заимствованы с пейпера
            self.s = 64.0 if not s else s
            self.m = 0.35 if not m else m
        self.loss_type = loss_type

    def forward(self, cosine, targ):
        cosine = cosine.clip(-1+1e-7, 1-1e-7) #32*500 - поставим ограничения на значения

        if self.loss_type == 'arcface':
            arcosine = cosine.arccos()        #32*500 - angle   
            M = F.one_hot(targ, num_classes = self.out_num) * self.m 
            arcosine += M                     #32*500
            cosine_corrected = arcosine.cos() * self.s          
        if self.loss_type == 'sphereface':                      
            arcosine = cosine.arccos()        #32*500 - angle   
            arcosine = arcosine * self.m                        
            cosine_corrected = arcosine.cos() * self.s          
        if self.loss_type == 'cosface':
            M = F.one_hot(targ, num_classes = self.out_num) * self.m 
            cosine -= M                       #32*500
            cosine_corrected = cosine * self.s                  
        return F.cross_entropy(cosine_corrected, targ) 