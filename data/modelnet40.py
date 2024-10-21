
import copy
from os import curdir
import torch
import torchvision
import numpy as np
from PIL import Image
import glob
import torch.utils.data
from torchvision import transforms


class ClientImgCollator(object):
    def __init__(self, local_views  ):
        self.local_views = local_views
        
    def __call__(self, batch):
        labels = []
        imgs = []
        for sample in batch:
            
            for i in range(len(self.local_views)):
                labels.append(sample[1])
                imgs.append(sample[0][i])
        
        return  torch.stack(imgs), torch.tensor(labels, dtype=torch.long) 
    

class ClientImgCollatorRandomLabel(object):        
    def __init__(self, local_views  ):
        self.local_views = local_views
        
    def __call__(self, batch):
        labels = []
        soft_labels=[]
        imgs = []
     
        for sample in batch:
            for i in range(len(self.local_views)):
                soft_labels.append(sample[1])
                labels.append(sample[2])
                imgs.append(sample[0][i])
        
        
        return  torch.stack(imgs), torch.stack(soft_labels), torch.tensor(labels, dtype=torch.long) 


def multi_view_path_split(root_dir,  have_val=False, num_models=0, num_views=12, shuffle=True, load_views=12):
    classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

    set_ = root_dir.split('/')[-1]
    parent_dir = root_dir.rsplit('/',2)[0]
    filepaths = []
   
    if have_val == False:
        for i in range(len(classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+ classnames[i]+'/'+set_+'/*.png'))
            ## Select subset for different number of views
            stride = int(12/num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                filepaths.extend(all_files)
            else:
                filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            filepaths = filepaths_new
        print("modelnet40 len filepaths for 12 views", len(filepaths))
        return filepaths
    else:
        valid_size=0.1
        val_filepaths =[]
        for i in range(len(classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+ classnames[i]+'/'+set_+'/*.png'))
            ## Select subset for different number of views
            stride = int(12/num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]
            if num_models!= 0:     # otherwise Use the whole dataset
                all_files = all_files[:min(num_models,len(all_files))]
            
            num_train = len(all_files)//num_views
            val_size = int(np.floor(valid_size * num_train))
            train_size = num_train - val_size
            # print("class {}, train {}, val {}".format(i, train_size, val_size))

            filepaths.extend(all_files[val_size*num_views:])
            val_filepaths.extend(all_files[:val_size*num_views])


        if shuffle==True:
            rand_idx = np.random.permutation(int(len(filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            filepaths = filepaths_new
            print("modelnet40 len filepaths for 12 views - train, val:", len(filepaths), len(val_filepaths))
            return filepaths, val_filepaths
        else:
            return filepaths, val_filepaths
          
      


class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, filepaths, test_mode=False, num_views=12, local_views=[]):
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.filepaths =filepaths 
        self.test_mode = test_mode
        self.num_views = num_views
        # self.load_views = load_views
        self.local_views = local_views

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.Resize([56, 56]),
                transforms.ToTensor(),
             
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.Resize([56, 56]),

                transforms.ToTensor(),
               
            ])
        
        self.targets = []
        for idx in range(int(len(self.filepaths)/self.num_views)): 
            path = self.filepaths[idx*self.num_views]
            class_name = path.split('/')[-3]
            class_id = self.classnames.index(class_name)
            self.targets.append(class_id)
        print(len(self.targets))
        


    def __len__(self):
        return int(len(self.filepaths)/self.num_views)


    def __getitem__(self, idx):

        path = self.filepaths[idx*self.num_views]

        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in self.local_views: # only load client's views - e.g., four
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB') # 3, 224, 224
            if self.transform:
                im = self.transform(im)
               
            imgs.append(im)

        return  torch.stack(imgs), class_id


class Data(object):
    def __init__(self, split, train=True, transform=None, batch_size=8, shuffle=True, nThreads=10,  randomized_label_privacy= None):
        self.data_loader = {}
        self.num_steps = 0
        self.nThreads = nThreads
        self.transform = transform
        self.batch_size = batch_size
        self.num_views= 12 # so we pick from 0,1,2,3,xxx view...
        self.num_obj = 1000 # for each category 

        n_models_train = self.num_obj * self.num_views 
        self.train = train 
        if train==True: 
            self.val_data_loader = {}
            self.val_dataset_list =[]
            self.train_dataset_list=[]
            train_filepaths, val_filepaths = multi_view_path_split('raw_data/modelnet40_images_new_12x/*/train',  have_val=True, num_models=n_models_train, num_views=self.num_views , shuffle=shuffle, load_views=split[-1])
            

        else:
            self.test_dataset_list=[]
            test_filepaths = multi_view_path_split('raw_data/modelnet40_images_new_12x/*/test',  have_val=False, num_models=0, num_views=self.num_views , shuffle=False, load_views=split[-1])
            
        # build data loader
        all_views_idx =list(range(12))

        train
        for uid in range(len(split)-1):
            local_views = all_views_idx[split[uid]:split[uid+1]]
            cli_collator = ClientImgCollator( local_views )

            if self.train:
                _val_dataset =  MultiviewImgDataset(val_filepaths,test_mode=True, num_views=self.num_views, local_views=local_views)
        
                self.val_data_loader[uid]=  torch.utils.data.DataLoader(_val_dataset,
                                                    batch_size=self.batch_size,  
                                                    collate_fn=cli_collator,
                                                    shuffle=False,
                                                    num_workers=self.nThreads)
                self.val_num_steps = len(self.val_data_loader[uid])
                self.val_dataset_list.append(_val_dataset)

                _train_dataset = MultiviewImgDataset(train_filepaths,test_mode=False, num_views=self.num_views, local_views=local_views)
                
                if randomized_label_privacy!=None:
                    random_label_cli_collator=  ClientImgCollatorRandomLabel( local_views )
                    from utils.alibi import NoisedDataset
                    _train_dataset = NoisedDataset(
                                _train_dataset, 40, randomized_label_privacy
                            )
                
                    print('NoisedDataset  train_ds',len(_train_dataset) )
                    self.data_loader[uid] = torch.utils.data.DataLoader(_train_dataset,
                                                        batch_size=self.batch_size,
                                                        collate_fn=random_label_cli_collator,
                                                        shuffle=False,
                                                        num_workers=0)
                else:
                    self.data_loader[uid] = torch.utils.data.DataLoader(_train_dataset,
                                                            batch_size=self.batch_size,
                                                            collate_fn=cli_collator,
                                                            shuffle=False,
                                                            num_workers=0)
                # update num_steps/per round
                self.num_steps = len(self.data_loader[uid])
                self.train_dataset_list.append(_train_dataset)
                self.train_size = len(_train_dataset)
  
                print("-> data in party:{}, train size: {} val size:{}, views: {}, train num_batches {} val num_batches {}".format(uid, len(self.train_dataset_list[0]), len(self.val_dataset_list[0]),local_views, self.num_steps, self.val_num_steps)) 

            else:
                _test_dataset = MultiviewImgDataset(test_filepaths,test_mode=True, num_views=self.num_views, local_views=local_views)
                
                self.data_loader[uid] = torch.utils.data.DataLoader(_test_dataset,
                                                        batch_size=self.batch_size,
                                                        collate_fn=cli_collator,
                                                        shuffle=False,
                                                        num_workers=0)
                # update num_steps/per round
                self.num_steps = len(self.data_loader[uid])
                self.test_dataset_list.append(_test_dataset)

                print("-> data in party:{}, test size:{}, views: {}, num_batches: {}".format(uid, len(self.test_dataset_list[0]),local_views, self.num_steps)) 
