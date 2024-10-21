import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math 


def clip_activation(x, max_norm=1,norm_type=2):
    total_norm = torch.norm(x, p=norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0) # only clip the value when total norm is larger than max norm
   
    y = x*clip_coef_clamped
    return y




class CIFAR_CNN(nn.Module):
    def __init__(self, emb_dim=60, dropout=0.0):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat =nn.Flatten(start_dim=1, end_dim=-1)  # size 128 

        self.fc1 = nn.Linear(128, emb_dim, bias=True)
 

        self.bn1 = nn.BatchNorm1d(emb_dim)

    
    def calc_representation(self, x):
        
        x = self.avg_pool(F.relu(self.conv1(x)))
        x = self.avg_pool(F.relu(self.conv2(x)))
        x = self.avg_pool(F.relu(self.conv3(x)))
        x = self.adapt_pool(F.relu(self.conv4(x)))
        x = F.relu(  self.bn1(self.fc1(self.flat(x))))
        # x = F.relu( self.fc1(self.flat(x)))
        return x


    def forward(self, x, vis=False ):
        x=  F.interpolate(x, (32, 32))
        x = self.calc_representation(x)

        return x


class ConMLP(nn.Module):
    def __init__(self, in_dim,  hid_dim=120, emb_dim=60):
        super(ConMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, emb_dim)
     
    def forward(self, x, vis=False):
        x=x.reshape(x.shape[0], -1)
        x_1  = F.relu(self.fc1(x))
        x_2 = F.relu(self.fc2(x_1))
        x= x_2 
        
        return x

def get_server_model(model_type, num_classes,  emb_dim, dropout_rate=0.25):
    if model_type== 'linear':
        model= LinearHead(num_classes, emb_dim)
    elif model_type=='nonliear2layer':
        model =  NonLinearHead2Layer(num_classes, emb_dim, dropout_rate)
    elif model_type=='nonliear3layer':
        model = NonLinearHead3Layer(num_classes, emb_dim, dropout_rate)
  
    
    return model 


class LinearHead(nn.Module):
    def __init__(self, num_classes, emb_dim=60):
        super(LinearHead, self).__init__()
        self.fc = nn.Linear(emb_dim, num_classes,bias=False)
    def forward(self, x):
        x = self.fc(x)
        return x


class NonLinearHead2Layer(nn.Module):
    def __init__(self, num_classes, emb_dim=60, dropout_rate=0.25):
        super(NonLinearHead2Layer, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.fc2 = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class NonLinearHead3Layer(nn.Module):
    def __init__(self, num_classes, emb_dim=60, dropout_rate=0.25):
        super(NonLinearHead3Layer, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.fc3 = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class WeightedAgg(nn.Module):
    def __init__(self,  num_clients):
        super(WeightedAgg, self).__init__()
        self.agg_weights = nn.Parameter(torch.ones(num_clients)/num_clients)
    
    def forward(self, x_list ):
        x = x_list[0] * self.agg_weights[0]
        for i in range(1,len(x_list)):
            x+= x_list[i] * self.agg_weights[i]
        return x
