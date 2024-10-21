import os
import json
import torch
from model.basic import LinearHead, WeightedAgg, get_server_model
from utils.utils import *
import math
import time
from torch.autograd import Variable
from opacus import PrivacyEngine

from utils.alibi import Ohm
import torch.nn.functional as f




class Coordinator(object):
    def __init__(self, conf, train_data,test_data,randomized_label_privacy= None ):
        self.conf = conf
        self.train_data = train_data
        self.train_num_steps= train_data.num_steps
        self.test_data= test_data
        self.test_num_steps= test_data.num_steps
        self.val_num_steps = train_data.val_num_steps
        self.randomized_label_privacy = randomized_label_privacy 

 
        self.fed_clients = self.conf.fed_clients
        self.party = self.conf.fed_vertical["party"]
        self.init()

    def init(self ):
        print("-> initial server")    
        self.num_clients= len(self.party)    
        # setup model
        for uid, party in enumerate(self.party):
            self.fed_clients[uid].model = create_network(party.num_features , self.conf)

            if len(self.conf.gpu_ids)>1:
                
                self.fed_clients[uid].model=  torch.nn.DataParallel(self.fed_clients[uid].model.to(self.conf.gpu_ids[0]),  device_ids=self.conf.gpu_ids) 
            else:

                self.fed_clients[uid].model= self.fed_clients[uid].model.to(self.conf.device)

            self.fed_clients[uid].optimizer =  create_local_optimizer(self.conf.local_optimizer,self.fed_clients[uid].model.parameters(), self.conf)


        self.linearhead=  get_server_model(self.conf.server_model_type, num_classes=self.conf.num_classes, emb_dim=self.conf.emb_dim, dropout_rate=self.conf.dropout_rate).to(self.conf.device)

        self.server_optimizer= create_optimizer(self.conf.optimizer, self.linearhead.parameters(), self.conf)

        self.weigted_agg= WeightedAgg( num_clients= len(self.party)).to(self.conf.device)
        self.weigted_agg_optimizer=  torch.optim.SGD(self.weigted_agg.parameters(),lr= self.conf.vafl_weightedagg_lr)  # 1e-5 is good ?
       
        self.dummy_model = LinearHead(num_classes=self.conf.num_classes, emb_dim=self.conf.emb_dim).to(self.conf.device)
        self.dummy_optimizer = create_optimizer(self.conf.optimizer, self.dummy_model.parameters(), self.conf)

        self.criterion= torch.nn.CrossEntropyLoss().to(self.conf.device)
   
        if self.randomized_label_privacy!= None:
            self.labeldp_criterion= Ohm(
                    privacy_engine=self.randomized_label_privacy,
                    post_process=self.conf.LabelDP_post_process,
                )
            
        print("lr {} bs {} ".format( self.conf.learning_rate, self.conf.batch_size))
        if self.conf.use_DP== True:
            print("Use DP: sigma {}, clip {} lr {}".format(self.conf.DP_sigma, self.conf.max_per_sample_clip_norm , self.conf.learning_rate))
            privacy_engine = PrivacyEngine(
                self.dummy_model,
                sample_rate= 1 ,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.conf.DP_sigma ,
                max_grad_norm=self.conf.max_per_sample_clip_norm,

            )
            privacy_engine.attach(self.dummy_optimizer)


    def run(self, val_log=1 ):
        best_acc = 0
        train_result_dict = {"loss": [], "acc": [], "epoch": [], "eps":[],"commu_round":[]}
        val_result_dict = {"loss": [], "acc": [], "epoch": [],"commu_round":[]}
        test_result_dict = {"loss": [], "acc": [], "epoch": [], "eps":[],"commu_round":[],"output_norm":[]}
    
        commu_round=0 
        for epoch in range(self.conf.num_round+1):
            time_epoch_start = time.time()
            if epoch>0: 
                total_sample = 0
                correct=0
                total_loss=0.0
                total_batch = 0
                if self.randomized_label_privacy!= None:
                    self.randomized_label_privacy.train()
                    assert isinstance(self.labeldp_criterion, Ohm) 
                # train 
                self.linearhead.train()
                self.weigted_agg.train()
                for uid, party in enumerate(self.party):
                    self.fed_clients[uid].model.train()
                    self.fed_clients[uid].prepare_dataloader_iter()

                for batch_idx in range(self.train_num_steps): # iter over batch 
                    commu_round+=1
                    # start round for all parties
                    emb_list= []
                    for uid, party in enumerate(self.party):
                        self.fed_clients[uid].prepare_batch()
                        self.fed_clients[uid].one_step_forward()
                        emb_list.append(self.fed_clients[uid].embedding)
                    agg_emb = self.weigted_agg(emb_list)
                    outputs = self.linearhead(agg_emb)
                    labels= self.fed_clients[0].y
                    if self.randomized_label_privacy!= None:
                        targets =  self.fed_clients[0].targets
                        loss = self.labeldp_criterion(outputs, targets) # noisy labels 
                    else:
                        loss = self.criterion(outputs, labels)  
                    _, predicted = torch.max(outputs.data, 1)
                    total_sample += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    ''' backward and optimize'''
                    ''' server '''
                    loss = loss / self.conf.accum_iter
                    loss.backward()
                    is_step_back = False
                    if ((batch_idx + 1) % self.conf.accum_iter == 0) or (batch_idx + 1 == self.train_num_steps):
                        is_step_back= True
                        self.server_optimizer.step()
                        self.weigted_agg_optimizer.step()
                        self.server_optimizer.zero_grad()
                        self.weigted_agg_optimizer.zero_grad()

                        if self.conf.use_DP== True:
                            self.dummy_optimizer.privacy_engine.steps+=1
                    total_loss+= loss.item()  * self.conf.accum_iter
                    total_batch += 1

                    ''' local worker '''
                    for uid, party in enumerate(self.party):
                        self.fed_clients[uid].one_step_backward(is_step_back)


                if self.conf.vis==True:
                    import wandb
                    wandb.log({
                        'LossTrain':total_loss/ total_batch,
                        'AccTrain':  float( 100 * correct / total_sample)
                    }, step=epoch)
                for uid, party in enumerate(self.party):
                    del self.fed_clients[uid].train_loader_iter
                print('Epoch %d  train loss: %.3f, train accuracy: %.2f %% , epoch time: %.4f ' % (epoch, total_loss/ total_batch,
                            100 * correct / total_sample ,time.time() - time_epoch_start ))
                train_result_dict["acc"].append(float( 100 * correct / total_sample ))
                train_result_dict["loss"].append(float(total_loss/ total_batch))
                train_result_dict["epoch"].append(epoch)
                with open(os.path.join(self.conf.output_path, "train_summary.json"), "w") as outfile:
                    json.dump(train_result_dict, outfile)

            if self.conf.use_DP== True:
                test_epsilon, test_best_alpha = self.dummy_optimizer.privacy_engine.get_privacy_spent(self.conf.DP_delta)
                if test_epsilon>self.conf.max_eps:
                    break


            if self.randomized_label_privacy!= None:
                self.randomized_label_privacy.eval()
            
            
            for name, param in self.weigted_agg.named_parameters():
                if  'agg_weights' in name:
                    print(name, param.data)
            
            if epoch% val_log == 0:
                current_acc, val_result_dict = self.evaluate(epoch, commu_round,  'val', val_result_dict)
                with open(os.path.join(self.conf.output_path, "{}_summary.json".format('val')), "w") as outfile:
                    json.dump(val_result_dict, outfile)
                
                if current_acc > best_acc:
                    best_acc=current_acc
                    if self.conf.save_model ==True: 
                        self.save_models(epoch, '')
                  
                if self.conf.save_model ==True and (epoch+1 in self.conf.saved_epochs): 
                    self.save_models(epoch, '_epoch{}'.format(epoch))

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
            self.linearhead.eval()
            self.weigted_agg.eval()
            for uid, party in enumerate(self.party):
                self.fed_clients[uid].model.eval()
                self.fed_clients[uid].prepare_dataloader_iter(mode)
            test_total, test_correct, test_all_loss =0, 0, 0
            output_norm=[]
            for _ in range(num_steps):
                emb_list= []
                for uid, party in enumerate(self.party):
                    self.fed_clients[uid].prepare_batch(mode)
                    self.fed_clients[uid].one_step_forward(is_train=False)
                    emb_list.append(self.fed_clients[uid].embedding)
                    if mode =='test':
                        output_norm.append(torch.norm(self.fed_clients[uid].embedding,p=2).item())

                agg_emb = self.weigted_agg(emb_list)
                outputs = self.linearhead(agg_emb)
                labels= self.fed_clients[0].y
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_all_loss+= loss.item()
                test_correct += (predicted == labels).sum().item()
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
        
    

    def save_models(self, epoch, name):
        for uid, party in enumerate(self.party):  
            _path = os.path.join(self.conf.model_path, "server{}{}.ckpt".format(uid, name )) 
            torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.linearhead.state_dict(),
                            'weigted_agg_state_dict': self.weigted_agg.state_dict(),
                            }, _path)
            _path = os.path.join(self.conf.model_path, "client{}{}.ckpt".format(uid,name ))                 
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.fed_clients[uid].model.state_dict(),
                }, _path)
        print("save to", _path)