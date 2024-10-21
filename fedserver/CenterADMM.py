import torch
import numpy as np
from scipy.optimize import minimize, minimize_scalar, root_scalar
from scipy import sparse
import math
import time

from multiprocessing import current_process, Pool 
pool = Pool()


def update_z_onebyone_para(args):
    z_i, onehot_label_i, dual_i,  s_Q_i, rho = args
    rest_args= (onehot_label_i, dual_i,  s_Q_i, rho)
    res = minimize(_z_obj_para(rest_args), z_i,  method="L-BFGS-B", jac=_z_jac_para(rest_args) , tol=1e-8,options={"maxiter":50,"disp":False})
    return res.x 

def _z_obj_para( args):
    onehot_label_i, dual_i,  s_Q_i, rho = args
    v=lambda z:  - np.dot(onehot_label_i,z.transpose()) + np.log(np.sum(np.exp(z))+ 1e-10)   - np.dot(dual_i.transpose(),z) + rho /2  * np.linalg.norm(s_Q_i - z)**2
    return v

def _z_jac_para( args):
    onehot_label_i, dual_i,  s_Q_i, rho = args
    v=lambda z: - onehot_label_i + np.exp(z)/(np.sum(np.exp(z))) - dual_i + rho *( z - s_Q_i)
    return v
 


class CenterADMM:
    def __init__(self,conf, train_size,  num_class):
        self.conf = conf      
        self.train_size= train_size
        self.num_class = num_class
        self.z = np.zeros((self.train_size,self.num_class))
        self.dual = np.zeros((self.train_size,self.num_class))
        
        self.num_iter = 0 
        self.batch_size = conf.batch_size 
        self.batch_s_Q = np.zeros((self.batch_size,self.num_class))
        self.onehot_label = np.zeros((self.batch_size,self.num_class))
        self.batch_z =  np.zeros((self.batch_size, self.num_class))
        self.batch_dual =  np.zeros((self.batch_size, self.num_class))
        self.batch_id = 0


    def update(self, batch_s_Q,batch_id):
        self.batch_id = batch_id
        self.sample_start_idx= self.batch_id*self.batch_size
        self.sample_end_idx= self.sample_start_idx+ batch_s_Q.shape[0]

        self.batch_s_Q = batch_s_Q
        self.batch_z = self.z[self.sample_start_idx: self.sample_end_idx]  
        self.batch_dual = self.dual[self.sample_start_idx: self.sample_end_idx]

    
        self.update_z_onebyone()
      
        self.update_dual()

        self.num_iter += 1

    def update_z_onebyone(self): # different z for different sample !! 
        if self.conf.no_parallel== False:
            p=Pool(self.conf.mutipro_thread_num)
            args_lis= [(self.batch_z[idx], self.onehot_label[idx], self.batch_dual[idx],  self.batch_s_Q[idx], self.conf.rho) for idx in range( self.batch_s_Q.shape[0])]
            res=p.map(update_z_onebyone_para,args_lis)  
            for i in range(self.batch_s_Q.shape[0]):
                self.z[self.sample_start_idx+ i] = res[i] 
            p.close()
            del p
        else:
            for i in range(self.batch_s_Q.shape[0]):
                self.cur_idx = i
                _each_z_start_time =time.time() 
                res = minimize(self.z_obj, self.batch_z[i], method=self.conf.x_solver, jac=self.z_jac, tol=1e-8,options={"maxiter":50,"disp":False})
                self.z[self.sample_start_idx+ i] = res.x
        
    def z_obj(self,z):
        i = self.cur_idx
        l = self.l_per(z, i) # calculate the loss for one sample 
        linear = np.dot(self.batch_dual[i].transpose(),z) 
        aug = np.linalg.norm(self.batch_s_Q[i] - z)**2
        obj= l - linear +  self.conf.rho /2  * aug
        return obj
    
    def z_jac(self,z):
        i = self.cur_idx 
        jac = -self.onehot_label[i]+ np.exp(z)/np.sum(np.exp(z)) - self.batch_dual[i] + self.conf.rho*( z - self.batch_s_Q[i])
        return jac

    def l_per(self, z, i): # calculate the loss for one sample 
        loss = - np.dot(self.onehot_label[i],z.transpose()) + np.log(np.sum(np.exp(z)))
        return loss 
    
        
    def update_dual(self):
        self.dual[self.sample_start_idx: self.sample_end_idx] = self.batch_dual + self.conf.rho * (self.batch_s_Q - self.z[self.sample_start_idx: self.sample_end_idx])
        
    def get_z_dual(self):
        return self.z[self.sample_start_idx: self.sample_end_idx], self.dual[self.sample_start_idx: self.sample_end_idx] 


