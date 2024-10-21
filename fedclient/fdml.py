import copy
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from model.basic import clip_activation

class MyRotationTransform(torch.nn.Module):
    """Rotate by the given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

class Client():
    def __init__(self, uid, conf, party, train_data_loader, test_data_loader, val_data_loader, debug=False):
        self.uid = uid
        self.conf = conf
        self.party = party
        self.data_point = -1
        self.train_data_loader = train_data_loader
        self.test_data_loader= test_data_loader
        self.val_data_loader = val_data_loader
        self.debug = debug
        self.model= None
        self.optimizer= None
        self.scheduler= None
      
    def prepare_dataloader_iter(self,mode='train'):
        if mode=='train':
            self.train_loader_iter = iter(self.train_data_loader)
        elif mode=='val':
            self.val_loader_iter = iter(self.val_data_loader)
        elif mode=='test':
            self.test_loader_iter = iter(self.test_data_loader)

    def prepare_batch(self, mode='train'):
        if mode=='train':
            if self.conf.use_LabelDP: 
                x, targets, y = next(self.train_loader_iter)
                self.targets = targets.to(self.conf.device)
            else:
                x, y = next(self.train_loader_iter)
        elif mode=='val':
            x, y = next(self.val_loader_iter)
        elif mode=='test':
            x, y = next(self.test_loader_iter)
     
        self.x, self.y = x.to(self.conf.device), y.to(self.conf.device)
    

    def one_step_forward(self,is_train=True ):
        self.local_tmp_logits= self.model(self.x)
        if is_train:
            if self.conf.use_DP==True:
                self.local_tmp_logits = clip_activation(self.local_tmp_logits, max_norm=self.conf.max_per_sample_clip_norm)
                noise = torch.cuda.FloatTensor( self.local_tmp_logits.shape, device= self.local_tmp_logits.device ).normal_(mean=0, std=self.conf.DP_sigma * self.conf.max_per_sample_clip_norm)
                self.local_tmp_logits.add_(noise)

            self.logits=  torch.autograd.Variable(self.local_tmp_logits.data, requires_grad=True)  
        else:
            self.logits=self.local_tmp_logits
 
    def one_step_backward(self, is_step_back =False):
        
        self.local_tmp_logits.backward(self.logits.grad) # accumate the grad
        if is_step_back:
            self.optimizer.step()
            self.optimizer.zero_grad()
     