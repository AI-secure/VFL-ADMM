import copy
import torch
import torch.nn.functional as F
import math
from scipy.optimize import minimize, minimize_scalar, root_scalar
from scipy import sparse
import numpy as np
from torch.autograd import Variable
import time 
from model.basic import clip_activation



class Client():
    def __init__(self, uid, conf, party, train_data_loader, test_data_loader,val_data_loader, train_size,num_class, batch_size):
        self.uid = uid
        self.conf = conf
        self.party = party
        self.data_point = -1
        self.train_data_loader = train_data_loader
        self.test_data_loader= test_data_loader
        self.val_data_loader = val_data_loader
        self.model= None
        self.optimizer= None
        self.scheduler= None
      
        self.train_size= train_size
        self.num_class = num_class
        self.batch_id= 0
        self.batch_size = batch_size

        # initialize the parameters
    
        self.hx = np.zeros((self.train_size,self.num_class)) # store
        self.batch_sum_hx = np.zeros((self.batch_size,self.num_class))
        self.batch_hx = np.zeros((self.batch_size,self.num_class))

       
        self.batch_z =  np.zeros((self.batch_size, self.num_class))
        self.batch_dual =  np.zeros((self.batch_size, self.num_class))

    def prepare_dataloader_iter(self,mode='train'):
        if mode=='train':
            self.train_loader_iter = iter(self.train_data_loader)
        elif mode=='val':
            self.val_loader_iter = iter(self.val_data_loader)
        elif mode=='test':
            self.test_loader_iter = iter(self.test_data_loader)

    def prepare_batch(self,batch_id=0, mode='train'):
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
        self.batch_id= batch_id
        self.sample_start_idx= self.batch_id*self.batch_size
        self.sample_end_idx= self.sample_start_idx+ self.x.shape[0]


    def one_step_forward(self,is_train=True):
        self.local_embed= self.model(self.x)
        self.logits= self.w(self.local_embed)

        if self.conf.use_DP==True and is_train==True:
            self.logits = clip_activation(self.logits, max_norm=self.conf.max_per_sample_clip_norm)
            noise = torch.cuda.FloatTensor( self.logits.shape, device= self.logits.device ).normal_(mean=0, std=self.conf.DP_sigma * self.conf.max_per_sample_clip_norm)
            self.logits.add_(noise)

        if is_train==True:
            self.hx[self.sample_start_idx: self.sample_end_idx] = self.logits.cpu().detach().numpy()
    
    def local_loss(self, hw, _ep, is_vis):
        dual_term =  torch.sum(torch.mul(hw,self.batch_dual ))
        temp_diff = self.batch_sum_hx_wo_k  + hw - self.batch_z
       
        temp_norm= torch.linalg.norm(temp_diff ,dim=1)  
        penalty_term=   torch.sum(torch.pow( temp_norm , 2))
        local_objective = (dual_term + self.conf.rho/2* penalty_term  ) / self.batch_z.shape[0]
        
        return local_objective

    def local_model_admm_update(self, batch_z,  batch_dual, batch_sum_hx_wo_k, local_epoch , is_vis =False ):
        
        self.batch_z = batch_z
        self.batch_dual = batch_dual
        self.batch_sum_hx_wo_k = batch_sum_hx_wo_k

        for _ep in range(local_epoch):
            
            _h = self.model(self.x) 
            assert not _h.isnan().any() 
            hw= self.w(_h)
            
            local_objective= self.local_loss(hw, _ep, is_vis)
           
            self.optimizer.zero_grad()
            self.w_optimizer.zero_grad()

            local_objective.backward()
       
            self.w_optimizer.step()
            self.optimizer.step()

  


    def get_hx(self, is_train=True):
        if True == is_train:
            return self.hx[self.sample_start_idx: self.sample_end_idx]
        else:
            return self.logits 
