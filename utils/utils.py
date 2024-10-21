import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

def in_class(predict, label, num_class=10):

    probs = []
    for i in range(num_class):
        in_class_id = torch.tensor(label == i, dtype= torch.float)
        correct_predict = torch.tensor(predict == label, dtype= torch.float)
        in_class_correct_predict = (correct_predict) * (in_class_id)
        if torch.sum(in_class_id)>0:
            acc = torch.sum(in_class_correct_predict).item() / torch.sum(in_class_id).item()
            probs.append(acc)

    return probs


def create_local_optimizer(choice, para, conf):
    
    if choice == "sgd_momentum":
        optimizer = optim.SGD(para,
                                lr=conf.local_learning_rate, momentum=conf.momentum,weight_decay=conf.local_weight_decay) 
    elif choice == "adam":
        optimizer = optim.Adam(para,
                                lr=conf.local_learning_rate, weight_decay=conf.local_weight_decay) 
    return optimizer

def create_optimizer(choice, para, conf):
    if choice == "sgd":
        optimizer = optim.SGD(para,
                                lr=conf.learning_rate, weight_decay=conf.weight_decay)
    elif choice == "sgd_momentum":
        optimizer = optim.SGD(para,
                                lr=conf.learning_rate, momentum=conf.momentum, weight_decay=conf.weight_decay)
    elif choice == "RMSprop":
        optimizer = optim.RMSprop(para,
                                lr=conf.learning_rate, weight_decay=conf.weight_decay)
    elif choice == "adagrad":
        optimizer = optim.Adagrad(para,
                                lr=conf.learning_rate, weight_decay=conf.weight_decay)
    elif choice == "adam":
        optimizer = optim.Adam(para,
                                lr=conf.learning_rate, weight_decay=conf.weight_decay)
    return optimizer

def create_network(num_features, conf):
    if conf.dataset == "mnist":
        from model.basic import ConMLP
        model = ConMLP(in_dim= num_features, emb_dim=conf.emb_dim )
    elif "cifar" in conf.dataset:
        from model.basic import CIFAR_CNN
        model = CIFAR_CNN( emb_dim=conf.emb_dim )       

    elif conf.dataset == "modelnet40":
        from model.svcnn import SVCNN
        model = SVCNN( emb_dim=conf.emb_dim)
    elif conf.dataset == "nus":
        from model.basic import ConMLP
        model = ConMLP(in_dim= num_features,  emb_dim=conf.emb_dim )

    return  model 



def create_whole_network(num_features, conf):
    if conf.dataset == "mnist" or conf.dataset == "nus":
        from model.wholemodel import ConMLP
        model = ConMLP(in_dim= num_features, emb_dim=conf.emb_dim , num_class = conf.num_classes )
    elif "cifar" in conf.dataset:
        from model.wholemodel import CIFAR_CNN
        model = CIFAR_CNN( emb_dim=conf.emb_dim )
    elif  conf.dataset == "modelnet40":
        from model.wholemodel import SVCNN
        model = SVCNN( emb_dim=conf.emb_dim)
    return  model 


def create_scheduler(optimizer,conf):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.milestones, gamma=conf.lr_gamma)
    return scheduler
    
def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm