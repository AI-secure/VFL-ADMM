import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models

class CIFAR_CNN(nn.Module):
    def __init__(self, emb_dim=60, dropout=0.0):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat =nn.Flatten(start_dim=1, end_dim=-1) 

        self.fc1 = nn.Linear(128, emb_dim, bias=True)



        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.linearhead = nn.Linear(emb_dim, 10 ,bias=False)

    
    # with one batch normalization in the last layer
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
        x=  self.linearhead(x)
        return x

class ConMLP(nn.Module):
    def __init__(self, in_dim,  hid_dim=120, emb_dim=60, num_class=10):
        super(ConMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, emb_dim)
        self.linearhead = nn.Linear(emb_dim, num_class ,bias=False)

    def forward(self, x, vis=False):
        x=x.reshape(x.shape[0], -1)
        x_1  = F.relu(self.fc1(x))
        x_2 = F.relu(self.fc2(x_1))
        x=  self.linearhead(x_2)
        
        return x


class SVCNN(nn.Module):

    def __init__(self,  emb_dim=60, pretraining=True, cnn_name='resnet18'):
        super(SVCNN, self).__init__()

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        # self.nclasses = nclasses
        self.emb_dim = emb_dim
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
       

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,emb_dim)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512,emb_dim)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048,emb_dim)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
            self.net_2._modules['6'] = nn.Linear(4096, emb_dim)
        self.bn1 = nn.BatchNorm1d(emb_dim)

        self.linearhead = nn.Linear(emb_dim, 40 ,bias=False)

    def forward(self, x):
        
        if self.use_resnet:
            feat=  self.net(x)
        else:
            y = self.net_1(x)

            feat = self.net_2(y.view(y.shape[0],-1))
        feat= F.relu(  self.bn1(feat))
        output = self.linearhead(feat)
        return output