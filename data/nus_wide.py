from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import os
from utils.nus_wide_data_util import get_labeled_data
import copy 
    
from torch.utils.data import random_split
class NUSDataset(data.Dataset):
   
    def __init__(self, is_train, num_samples):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.train = is_train
        self.top_k= ['buildings', 'grass', 'animal', 'water', 'person'] 
        
        if is_train:
            X_image, X_text, Y = get_labeled_data('raw_data', self.top_k, num_samples, 'Train')
        else:
            X_image, X_text, Y = get_labeled_data('raw_data', self.top_k, num_samples, 'Test')
        X_image = np.array(X_image).astype('float32')
        X_text = np.array(X_text).astype('float32')
        Y = np.array(Y).astype('int')
        Y= np.argmax(Y,axis=1)
        X= np.concatenate((X_image, X_text), axis=1)
        self.data = X
        self.targets = Y 


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data, label = self.data[index], self.targets[index]

        return data, label

class Data(object):
    def __init__(self, split, train=True, transform=None, batch_size=640, shuffle=True, nThreads=10,valid_size=0.1, randomized_label_privacy= None):
        self.data_loader = {}
        self.num_steps = 0
        self.nThreads = nThreads
        self.transform = transform
        self.batch_size = batch_size
        self.train = train
        if train:
            self.dataset = NUSDataset(is_train=True, num_samples=60000)
            self.val_data_loader = {}
            self.valid_size= valid_size
        else:
            self.dataset = NUSDataset(is_train=False, num_samples=10000)

        if shuffle:
            size = len(self.dataset)
            idx = np.random.choice(size, size, replace=False)
            self.dataset.data = self.dataset.data[idx]
            self.dataset.targets = self.dataset.targets[idx]
        # split data into multiple parts
        num_train = len(self.dataset)
        datasets = self.split_vertical(self.dataset, split=split) 

        if self.train == False:
        
            for uid in range(len(datasets)):
                dataset= datasets[uid]
                print("-> data in party:{}, size:{} batch_size {} ".format(uid, len(dataset), self.batch_size ))        
                self.data_loader[uid] = torch.utils.data.DataLoader(dataset,
                                                    batch_size=self.batch_size, 
                                                    shuffle=False,
                                                    num_workers=self.nThreads)
                # update num_steps/per round
                self.num_steps = len(self.data_loader[uid])


        else:
            self.val_data_loader = {}
            val_size = int(np.floor(valid_size * num_train))
            self.train_size = num_train - val_size

            
            for uid in range(len(datasets)):
                dataset= datasets[uid]
               
                indices = list(range(num_train))
                train_idx, valid_idx = indices[val_size:], indices[:val_size]
                train_ds= copy.deepcopy(dataset)
                train_ds.data = dataset.data[train_idx] # 
                train_ds.targets = [dataset.targets[index] for index in train_idx] 

                val_ds= copy.deepcopy(dataset)
                val_ds.data = dataset.data[valid_idx] # 
                val_ds.targets = [dataset.targets[index] for index in valid_idx] 

                if randomized_label_privacy!=None:
                    from utils.alibi import NoisedDataset
                    train_ds = NoisedDataset(
                                train_ds, 5, randomized_label_privacy
                            )
                
                    print('NoisedDataset, train_ds',len(train_ds), 'val_ds', len(val_ds) )

                
                self.data_loader[uid] = torch.utils.data.DataLoader(train_ds,
                                                    batch_size=self.batch_size, 
                                                    shuffle=False,
                                                    num_workers=self.nThreads)
                self.val_data_loader[uid]=  torch.utils.data.DataLoader(val_ds,
                                                    batch_size=self.batch_size,  
                                                    shuffle=False,
                                                    num_workers=self.nThreads)
                print("-> data in party:{}, train loader size:{} val loader size {} batch_size {}".format(uid, len(self.data_loader[uid]), len(self.val_data_loader[uid]) ,self.batch_size ))
                self.val_num_steps = len(self.val_data_loader[uid])
 
                self.num_steps = len(self.data_loader[uid])
        

    def split_vertical(self, dataset, split=[0,14, 28]):
        datasets= []

        for idx in range(0, len(split)-1):
            temp_dataset = copy.deepcopy(dataset)
            temp_dataset.data = temp_dataset.data[:,  split[idx]:split[idx+1]]
            datasets.append(temp_dataset)

        for idx in range(len(datasets)):
            print(datasets[idx].data.shape)
        
        return datasets

   