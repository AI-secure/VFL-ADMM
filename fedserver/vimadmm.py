import os
import json
from random import random
import torch
import numpy as np
from torch.autograd import Variable
from fedserver.CenterADMM import CenterADMM
import math
import time
from utils.utils import *
from model.basic import LinearHead ,get_server_model
from utils.alibi import Ohm
from opacus import PrivacyEngine
import torch.nn.functional as f



def ce_loss_soft(output, target):

    output = f.softmax(output, dim=-1)
    output = output + 1e-10
    output = -torch.log(output)
    return (target * output).sum(dim=-1).mean()


class Coordinator(object):
    def __init__(self, conf, train_size, train_num_steps  , test_num_steps, val_num_steps, num_class, warm_up_path=None,randomized_label_privacy=None):
        self.conf = conf      
        self.train_size= train_size
        self.train_num_steps= train_num_steps
        self.test_num_steps= test_num_steps
        self.val_num_steps = val_num_steps
        self.fed_clients = self.conf.fed_clients
        self.num_class= num_class
        self.party = self.conf.fed_vertical["party"]
     
        self.warm_up_path = warm_up_path 
        self.randomized_label_privacy = randomized_label_privacy 
        
        print("-> initial server")
    
        # setup model
        for uid, party in enumerate(self.party):
            self.fed_clients[uid].model = create_network(party.num_features , self.conf)
            self.fed_clients[uid].w = get_server_model(self.conf.server_model_type, num_classes=self.conf.num_classes, emb_dim=self.conf.emb_dim, dropout_rate=self.conf.dropout_rate)

            if len(self.conf.gpu_ids)>1:
                self.fed_clients[uid].model=  torch.nn.DataParallel(self.fed_clients[uid].model.to(self.conf.gpu_ids[0]),  device_ids=self.conf.gpu_ids) 
                self.fed_clients[uid].w=  torch.nn.DataParallel(self.fed_clients[uid].w.to(self.conf.gpu_ids[0]),  device_ids=self.conf.gpu_ids) 
            else:
                self.fed_clients[uid].model= self.fed_clients[uid].model.to(self.conf.device)
                self.fed_clients[uid].w=  self.fed_clients[uid].w.to(self.conf.device)
            
            self.fed_clients[uid].optimizer =  create_local_optimizer(self.conf.local_optimizer,self.fed_clients[uid].model.parameters(), self.conf)
           
            self.fed_clients[uid].w_optimizer = create_optimizer(self.conf.optimizer, self.fed_clients[uid].w.parameters(), self.conf)
            self.fed_clients[uid].w_scheduler= create_scheduler(self.fed_clients[uid].w_optimizer,self.conf)
        
        self.dummy_model = LinearHead(num_classes=self.conf.num_classes, emb_dim=self.conf.emb_dim).to(self.conf.device)
        self.dummy_optimizer = create_optimizer(self.conf.optimizer, self.dummy_model.parameters(), self.conf)

        self.criterion= torch.nn.CrossEntropyLoss().to(self.conf.device)
        if self.randomized_label_privacy!= None:
            self.labeldp_criterion= Ohm(
                    privacy_engine=self.randomized_label_privacy,
                    post_process=self.conf.LabelDP_post_process,
                )

        self.max_iter= self.conf.max_iter
        print("lr {} bs {} local admm steps {}".format( self.conf.learning_rate, self.conf.batch_size, self.conf.local_admm_epoch))
        
        if self.conf.use_DP== True:
            print("Use DP: sigma {}, clip {} lr {}".format(self.conf.DP_sigma, self.conf.max_per_sample_clip_norm , self.conf.learning_rate))
            privacy_engine = PrivacyEngine(
                self.dummy_model,
                sample_rate= 1,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.conf.DP_sigma ,
                max_grad_norm=self.conf.max_per_sample_clip_norm,

            )
            privacy_engine.attach(self.dummy_optimizer)

    
    def load_model(self, model_dir):
        if os.path.isdir(model_dir):
            print("load model from", model_dir)
        
            name=""
            for uid, party in enumerate(self.party):  
                    model_path = os.path.join(model_dir, "server{}{}.ckpt".format(uid, name )) 
                    checkpoint = torch.load(model_path)
                    self.fed_clients[uid].w.load_state_dict(checkpoint['model_state_dict'])
            
                    model_path = os.path.join(model_dir, "client{}{}.ckpt".format(uid,name ))     
                    checkpoint = torch.load(model_path)
                    self.fed_clients[uid].model.load_state_dict(checkpoint['model_state_dict'])            
                    
                    self.fed_clients[uid].model.to(self.conf.device)
                    self.fed_clients[uid].w.to(self.conf.device)
                    self.fed_clients[uid].model.eval()
                    self.fed_clients[uid].w.eval()
            print("load model done")

    
    def get_one_hot(self, labels):
        labels = labels.cpu().detach().numpy()

        n=labels.shape[0]
        newC=np.zeros((n,self.num_class))
        for i in range(n):
            newC[i,labels[i]]=1.
        return newC
        

    def run(self,  val_log=1 ):
     
        train_result_dict = {"loss": [], "acc": [], "epoch": [], "eps":[],"commu_round":[]}
        val_result_dict = {"loss": [], "acc": [], "epoch": [],"commu_round":[]}
        test_result_dict = {"loss": [], "acc": [], "epoch": [], "eps":[],"commu_round":[],"output_norm":[]}
    
        commu_round=0 
            
        s_Q = np.zeros((self.train_size,self.num_class)) # sum h_k w_k  # store
        z =np.zeros((self.train_size,self.num_class)) # store
        dual =np.zeros((self.train_size,self.num_class)) # store
        centerADMM = CenterADMM(self.conf, self.train_size, self.num_class)
      
        best_acc= 0
        for epoch in range(self.conf.num_round+1):
            time_epoch_start = time.time()
            if epoch>0:

                total_sample = 0
                correct=0
                total_train_loss= 0
                total_batch =0 

                if self.randomized_label_privacy!= None:
                    self.randomized_label_privacy.train()
                    assert isinstance(self.labeldp_criterion, Ohm) 

                # train 
                for uid, party in enumerate(self.party):
                    self.fed_clients[uid].prepare_dataloader_iter()

                for batch_id in range(self.train_num_steps): 
                    commu_round+=1
                    for uid, party in enumerate(self.party):
                        self.fed_clients[uid].prepare_batch(batch_id)
                        self.fed_clients[uid].one_step_forward()
                    # just to track the  training loss and accuracy! 
        
                    out_list= []
                    for uid, party in enumerate(self.party):
                        out_list.append(self.fed_clients[uid].logits)
                    outputs = torch.sum(torch.stack(out_list),dim=0) # output : [batch_size, output_size]
                    labels= self.fed_clients[0].y

                    if self.randomized_label_privacy!= None:
                        targets =  self.fed_clients[0].targets
                        total_train_loss+= ce_loss_soft(outputs, targets).item() # noisy lables 
                    else:
                        total_train_loss+= self.criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_sample += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    total_batch+=1
    
            
                    #  Use autograd to compute the backward pass.
                    ''' server ADMM Solver '''
                    # set the labels for ADMM in the server side
                    if self.randomized_label_privacy!= None:
                        centerADMM.onehot_label =  self.fed_clients[0].targets.cpu().numpy()
                    else:
                        centerADMM.onehot_label = self.get_one_hot(self.fed_clients[0].y)

                    sample_start_idx= batch_id*self.conf.batch_size
                    sample_end_idx= sample_start_idx+ self.fed_clients[0].x.shape[0]

                    batch_z= z[sample_start_idx:sample_end_idx]
                    batch_dual = dual [sample_start_idx:sample_end_idx]

                    Q = []
                    for uid, party in enumerate(self.party):
                        Q.append(self.fed_clients[uid].get_hx(is_train = True))
                    batch_s_Q = np.sum(Q, axis=0) # sum h_k w_k
                    assert not np.any(np.isnan(batch_s_Q))
                    if self.randomized_label_privacy!= None:
                        # post processing for the targets
                        centerADMM.onehot_label  = self.labeldp_criterion.soft_target(torch.from_numpy(batch_s_Q).type(torch.FloatTensor).to(self.conf.device),self.fed_clients[0].targets).cpu().numpy()

                    centerADMM.update(batch_s_Q,batch_id) # to be update
                    batch_z, batch_dual = centerADMM.get_z_dual() # z,dual will be used by workers
                

                    z[sample_start_idx:sample_end_idx]= batch_z
                    dual[sample_start_idx:sample_end_idx] = batch_dual
                    s_Q[sample_start_idx:sample_end_idx] = batch_s_Q

                    batch_z_torch = torch.from_numpy(batch_z).type(torch.FloatTensor).to(self.conf.device)
                    batch_dual_torch = torch.from_numpy(batch_dual).type(torch.FloatTensor).to(self.conf.device)
                
                    for uid, party in enumerate(self.party):
        
                        _is_vis= True if uid == 0 else False
                        # set local models as training here
                        self.fed_clients[uid].model.train()
                        self.fed_clients[uid].w.train()
                        batch_s_Q_torch_wo_k = torch.from_numpy(batch_s_Q -  Q[uid]).type(torch.FloatTensor).to(self.conf.device)
                        self.fed_clients[uid].local_model_admm_update( batch_z_torch,  batch_dual_torch, batch_s_Q_torch_wo_k, local_epoch = self.conf.local_admm_epoch, is_vis =_is_vis ) 

                    if self.conf.use_DP== True:
                        self.dummy_optimizer.privacy_engine.steps+=1
                    
                for uid, party in enumerate(self.party):
                    del self.fed_clients[uid].train_loader_iter

                if self.conf.vis==True:
                    import wandb
                    wandb.log({
                        'LossTrain':total_train_loss/ total_batch,
                        'AccTrain':  float( 100 * correct / total_sample)
                    }, step=epoch)

                print('Epoch %d  train loss: %.3f, train accuracy: %.2f %% , epoch time: %.4f ' % (epoch, total_train_loss/ total_batch,
                            100 * correct / total_sample ,time.time() - time_epoch_start ))
                train_result_dict["acc"].append(float( 100 * correct / total_sample ))
                train_result_dict["loss"].append(float(total_train_loss/ total_batch))
                train_result_dict["epoch"].append(epoch)
                with open(os.path.join(self.conf.output_path, "train_summary.json"), "w") as outfile:
                    json.dump(train_result_dict, outfile)

            if self.conf.use_DP== True:
                test_epsilon, test_best_alpha = self.dummy_optimizer.privacy_engine.get_privacy_spent(self.conf.DP_delta)
                if test_epsilon>self.conf.max_eps:
                    break

            if self.randomized_label_privacy!= None:
                self.randomized_label_privacy.eval()
                
            # val 
            if epoch% val_log == 0:
                current_acc, val_result_dict = self.evaluate(epoch, commu_round,  'val', val_result_dict)
                with open(os.path.join(self.conf.output_path, "{}_summary.json".format('val')), "w") as outfile:
                    json.dump(val_result_dict, outfile)
                
                if current_acc > best_acc:
                    best_acc=current_acc
                    if self.conf.save_model ==True: 
                        self.save_models(epoch, '',z, dual)
                       
                if self.conf.save_model ==True and (epoch+1 in self.conf.saved_epochs): 
                    self.save_models(epoch, '_epoch{}'.format(epoch),z, dual)

                drop_threshold = self.conf.DP_drop_th if self.conf.use_DP== True else self.conf.drop_th
                if current_acc*100 < best_acc*100 -drop_threshold:
                    break
                
                # test 
                test_acc, test_result_dict = self.evaluate(epoch, commu_round,  'test', test_result_dict)
                if self.conf.use_DP== True:
                    test_result_dict["eps"].append(test_epsilon)
                    print('Epoch %d  test accuracy: %.2f %% DP Eps: %.2f, Alpha: %.2f, epoch entire time: %.4f ' % (epoch,test_acc *  100 , test_epsilon,  test_best_alpha,  time.time()- time_epoch_start )) 
                    import wandb
                    wandb.log({'EpsilonTest': test_epsilon,
                                'AlphaTest':  test_best_alpha}, step=epoch)
                with open(os.path.join(self.conf.output_path, "{}_summary.json".format('test')), "w") as outfile:
                    json.dump(test_result_dict, outfile)


        if self.conf.vis==True:
            wandb.finish()

    def evaluate(self, epoch, commu_round,  mode, result_dict):
        num_steps = self.val_num_steps if mode =='val' else self.test_num_steps
        with torch.no_grad():
            for uid, party in enumerate(self.party):
                self.fed_clients[uid].model.eval()
                self.fed_clients[uid].w.eval()
                self.fed_clients[uid].prepare_dataloader_iter(mode)
            test_total, test_correct, test_all_loss =0, 0, 0
            output_norm=[]
            all_label=[]
            all_pred=[]

            for _ in range(num_steps):
                out_list= []
                for uid, party in enumerate(self.party):
                    self.fed_clients[uid].prepare_batch(mode=mode)
                    self.fed_clients[uid].one_step_forward(is_train=False)
                    out_list.append(self.fed_clients[uid].logits)
                    if mode =='test':
                        if self.conf.method =='admm' or self.conf.method =='admm_detach':
                            output_norm.append(torch.norm(self.fed_clients[uid].local_embed,p=2).item())
                        elif self.conf.method == 'admmjoint':
                            output_norm.append(torch.norm(self.fed_clients[uid].logits,p=2).item())
                # \sum_k h_k w_k 
                outputs = torch.sum(torch.stack(out_list),dim=0)
                labels= self.fed_clients[0].y
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_all_loss+= loss.item()
                test_correct += (predicted == labels).sum().item()
                
                all_label.append(labels)
                all_pred.append(predicted)
           
            all_label = torch.cat(all_label).flatten()
            all_pred = torch.cat(all_pred).flatten()
            
            all_class_acc= in_class(all_label, all_pred, num_class=self.conf.num_classes)
            print(all_class_acc)
            std_class_acc= np.std(all_class_acc)
            mean_class_acc= np.mean(all_class_acc)
            normal_std_class_acc= std_class_acc/ mean_class_acc

            print(self.conf.model_path.split('/')[-2], mode, 'accuracy std: %.2f %%, mean: %.2f %%, normalized std: %.2f %%' % (std_class_acc*100, mean_class_acc*100, normal_std_class_acc*100 )) 
    

            if self.conf.vis==True:
                import wandb
                wandb.log({
                    'Loss{}'.format(mode): float(test_all_loss/num_steps),
                    'Acc{}'.format(mode):  float(100 * test_correct / test_total)}, step=epoch)
                if mode =='test':
                    wandb.log({
                        'AvgOutputNorm{}'.format(mode): sum(output_norm)/len(output_norm)
                    }, step=epoch)
                    

            print(self.conf.model_path.split('/')[-2], mode, 'Epoch %d  loss: %.3f, accuracy: %.2f %%' % (epoch, test_all_loss/num_steps,
                100 * test_correct / test_total ))
            result_dict["acc"].append(float( 100 * test_correct / test_total))
            result_dict["loss"].append(float(test_all_loss/num_steps))
            result_dict["epoch"].append(epoch)
            result_dict["commu_round"].append(commu_round)
            if mode =='test':
                result_dict["output_norm"].append(sum(output_norm)/len(output_norm))


        current_acc =  test_correct / test_total 
        
        return current_acc, result_dict
        
    

    def save_models(self, epoch, name, z, dual):
        for uid, party in enumerate(self.party):  
            _path = os.path.join(self.conf.model_path, "server{}{}.ckpt".format(uid, name )) 
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.fed_clients[uid].w.state_dict(),
                }, _path)
            _path = os.path.join(self.conf.model_path, "client{}{}.ckpt".format(uid,name ))                 
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.fed_clients[uid].model.state_dict(),
                }, _path)
        _path = os.path.join(self.conf.model_path, "z{}.ckpt".format(name))
        np.save(_path, z) 
        _path = os.path.join(self.conf.model_path, "dual{}.ckpt".format(name))
        np.save(_path, dual)
        print("save to", _path)

