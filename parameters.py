import argparse
import copy
import random
DATASETS = ['mnist','cifar','nus', 'modelnet40']
NUM_CLASS= {'mnist':10, 'cifar':10, 'nus':5, 'modelnet40':40}
import torch
import numpy as np
import datetime
import os 

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("set random seed", seed )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=300) # 100 200 300
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--method',type=str,default= 'vimsgd',
                choices=['vimsgd_concat', 'vafl', 'fdml',  'admm' , 'admmjoint','fedbcd','central'])

    # model 
 
    parser.add_argument('--server_model_type',type=str,default= 'linear',
                choices=['linear', 'nonliear2layer', 'nonliear3layer',])
    parser.add_argument('--dropout_rate',type=float,default=0.25)
    parser.add_argument('--emb_dim',type=int,default=60)
    # dataset  
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        default='mnist',
                        required=True)
    parser.add_argument('--features_split', nargs='+', type=int, default=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28])
    parser.add_argument('--real_batch_size',type=int,default=1024)

    parser.add_argument('--test_batch_size',type=int,default=128) 
    parser.add_argument('--nThreads',type=int,default=5)
    parser.add_argument('--num_round',type=int,default=30)
    parser.add_argument('--use_longtail', action='store_true' )

    
       
    # optimizer 
    parser.add_argument('--learning_rate',type=float,default=0.01)
    parser.add_argument('--vafl_weightedagg_lr',type=float,default=0.01)
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--lr_gamma',type=float,default=0.1)
    parser.add_argument('--milestones', nargs='+', type=int, default=[200])
    parser.add_argument('--weight_decay',type=float,default=0.001)
    parser.add_argument('--optimizer',type=str,default= 'sgd',
                choices=['sgd', 'sgd_momentum', 'RMSprop',  'adagrad' , 'adam'])
    parser.add_argument('--local_optimizer',type=str,default= 'sgd_momentum',choices=['sgd_momentum', 'adam'])
    parser.add_argument('--local_weight_decay',type=float,default=0.001)
    parser.add_argument('--warm_up_path',type=str, default='')
    parser.add_argument('--warm_up_epoch',type=int,default=0)

    # evaluation
    parser.add_argument('--load_model_dir',type=str, default='')
    parser.add_argument('--eval_only', action='store_true' )


    # DP 
    parser.add_argument('--use_DP', action='store_true' )
    parser.add_argument('--DP_sigma',type=float,default= 30)
    parser.add_argument('--max_per_sample_clip_norm',type=float,default= 0.005)
    parser.add_argument('--DP_delta',type=float,default= 1e-5)
    parser.add_argument('--max_eps',type=float,default= 15)

    parser.add_argument('--DP_drop_th',type=float,default= 20)
    parser.add_argument('--drop_th',type=float,default= 10)
    

    # ADMM 
    parser.add_argument('--rho',type=float,default=2)
    parser.add_argument('--x_solver',type=str,default= 'L-BFGS-B',choices=['L-BFGS-B', 'Newton-CG'])
    parser.add_argument('--no_parallel', action='store_true')
    parser.add_argument('--mutipro_thread_num',type=int,default=10)
    parser.add_argument('--local_admm_epoch',type=int,default=1)
    
    
    
    parser.add_argument('--save_model', action='store_true' )
    parser.add_argument('--saved_epochs', nargs='+', type=int, default=[150])
    parser.add_argument('--out_dir',type=str,default= 'iclr_debug')
    parser.add_argument('--scope_name',type=str,default= 'myexp')

    parser.add_argument('--vis', action='store_true' )
    parser.add_argument('--val_log',type=int,default=1)

    parser.add_argument('--wandb_key', default='',  type=str,   help='API key for W&B.')
    parser.add_argument('--project', default='VIMADMM', type=str,   help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')

    # explainability 
    parser.add_argument('--denoise_func', action='store_true' )
    parser.add_argument('--noisy_client', nargs='+', type=int, default=[])
    parser.add_argument('--noisy_input_level',type=float,default=0)


    parser.add_argument('--use_LabelDP', action='store_true' )
    parser.add_argument('--LabelDP_sigma',type=float,default= 2.0)
    parser.add_argument('--LabelDP_post_process',type=str,default= 'MAPWithPrior', 
                           choices=['MinMax', 'SoftMax', 'MinProjection', 'MAP', 'MAPWithPrior', 'RandomizedResponse'])
    parser.add_argument('--LabelDP_mechanism',type=str,default= 'Laplace',choices=['Laplace', 'Gaussian'])

    parser.add_argument('--contrast_wt',type=float,default= 0.5)

    parser.add_argument('--subsample_ratio',type=float,default= 1)

    parser.add_argument('--max_iter',type=int,default=1)
    parser.add_argument('--grad_threshold',type=float,default=0.02)

  
    args = parser.parse_args()
    init_seed(args.seed)
    format_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
  
    if torch.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu_ids[0])
    else:
        args.device = 'cpu'
    args.local_learning_rate  = args.learning_rate
    args.batch_size = args.real_batch_size
    args.accum_iter = args.real_batch_size//args.batch_size
    args.num_classes = NUM_CLASS[args.dataset]

   
    
    args.output_root =  os.path.join(args.out_dir,  args.scope_name)
    args.output_path =  os.path.join(args.out_dir,  args.scope_name, 'output{}'.format(args.seed))
    args.model_path =  os.path.join(args.out_dir,  args.scope_name, 'model{}'.format(args.seed))

    for path in [args.output_root, args.output_path, args.model_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            
    return args