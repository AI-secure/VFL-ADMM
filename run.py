from torchvision import datasets, transforms
from parameters import parse_args
import os 

def set_dataset(conf,randomized_label_privacy):
    # load splitted dataset
    if conf.dataset == "mnist":
        from data.mnist import Data

        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    elif conf.dataset == "cifar":
        from data.cifar import Data
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif conf.dataset == "modelnet40":
        from data.modelnet40 import Data
        train_transform = None
    elif conf.dataset == "nus":
        from data.nus_wide import Data
        train_transform = None


    train_data = Data(split=conf.features_split, transform=train_transform, batch_size=conf.batch_size, train=True,nThreads=conf.nThreads, randomized_label_privacy=randomized_label_privacy)
    test_data = Data(split=conf.features_split, transform=train_transform, batch_size=conf.batch_size, train=False,nThreads=conf.nThreads )
    return train_data, test_data


class Party():
    def __init__(self, uid, num_feature=10):
        self.uid = uid
        self.num_features = num_feature
       
def set_labelDP(conf):
    randomized_label_privacy = None 
    if conf.use_LabelDP:
        from utils.alibi import RandomizedLabelPrivacy
        randomized_label_privacy = RandomizedLabelPrivacy(
            sigma=conf.LabelDP_sigma,
            delta=1e-5 ,
            mechanism=conf.LabelDP_mechanism,
            device=None
        )
        randomized_label_privacy.increase_budget() # for noisy dataset
        epsilon, alpha = randomized_label_privacy.privacy
        print("label DP eps", epsilon, "alpha", alpha)
    return randomized_label_privacy



def main():  
    conf = parse_args()
    if conf.vis:
        import wandb
        _ = os.system('wandb login {}'.format(conf.wandb_key))
        os.environ['WANDB_API_KEY'] = conf.wandb_key
        wandb.init(project=conf.project, name=os.path.basename(conf.output_root) )
        wandb.config.update(conf)

    print(conf) 
    _partylist=[]
    for i in range(len(conf.features_split)-1):
        start, end  = conf.features_split[i], conf.features_split[i+1]
        if conf.dataset=='mnist':
            _partylist.append(Party(uid=i, num_feature=28* (end-start)) )
        else:
            _partylist.append(Party(uid=i, num_feature=(end-start)) )
    
    conf._partylist= _partylist
    conf.fed_vertical= { "party": conf._partylist}
    conf.fed_clients = {}

    randomized_label_privacy= set_labelDP(conf)
    train_data, test_data =set_dataset(conf,randomized_label_privacy)
    
    # setup client 

    if conf.method =='vafl' or conf.method =='vimsgd_concat':
        from fedclient.client import Client
        for uid, party in enumerate(conf.fed_vertical["party"]):
            conf.fed_clients[uid] = Client(uid, conf, party, train_data.data_loader[uid], test_data.data_loader[uid] , train_data.val_data_loader[uid])
    elif conf.method =='fdml':
        from fedclient.fdml import Client
        for uid, party in enumerate(conf.fed_vertical["party"]):
            conf.fed_clients[uid] = Client(uid, conf, party, train_data.data_loader[uid], test_data.data_loader[uid] , train_data.val_data_loader[uid] )
    elif conf.method =='fedbcd':
        from fedclient.fedbcd import Client
        for uid, party in enumerate(conf.fed_vertical["party"]):
            conf.fed_clients[uid] = Client(uid, conf, party, train_data.data_loader[uid], test_data.data_loader[uid] , train_data.val_data_loader[uid] )

    elif conf.method =='admm':
        from fedclient.admmsplit import Client 
        for uid, party in enumerate(conf.fed_vertical["party"]):
            conf.fed_clients[uid] = Client(uid, conf, party, train_data.data_loader[uid], test_data.data_loader[uid], train_data.val_data_loader[uid], train_data.train_size  , conf.num_classes, conf.batch_size )
   
    elif conf.method =='admmjoint':
        from fedclient.admmjoint import Client
        for uid, party in enumerate(conf.fed_vertical["party"]):
            conf.fed_clients[uid] = Client(uid, conf, party, train_data.data_loader[uid], test_data.data_loader[uid], train_data.val_data_loader[uid], train_data.train_size  , conf.num_classes, conf.batch_size )

  
    # setup server 
    if conf.method =='vimsgd_concat':
        from fedserver.vimsgd_concat import Coordinator as Server
        server = Server(conf, train_data, test_data,randomized_label_privacy=randomized_label_privacy)
    elif conf.method =='vafl':
        from fedserver.vafl import Coordinator as Server
        server = Server(conf, train_data, test_data,randomized_label_privacy=randomized_label_privacy)
    elif conf.method == 'fdml':
        from fedserver.fdml import Coordinator as Server
        server = Server(conf, train_data, test_data,randomized_label_privacy=randomized_label_privacy)
    elif conf.method =='fedbcd':
        from fedserver.fedbcd import Coordinator as Server
        server = Server(conf, train_data, test_data,randomized_label_privacy=randomized_label_privacy)

    elif conf.method =='admm' or conf.method =='admmjoint':
        from fedserver.vimadmm import Coordinator as Server
        _warm_up_path = os.path.join(conf.warm_up_path, 'model' )  if (conf.warm_up_epoch> 0 and conf.warm_up_path!='') else None 
        server = Server(conf, train_data.train_size , train_data.num_steps, test_data.num_steps, train_data.val_num_steps, conf.num_classes, warm_up_path= _warm_up_path , randomized_label_privacy=randomized_label_privacy)
   
    

    if conf.eval_only:
        server.load_model(conf.load_model_dir)
        server.evaluate(epoch=0, commu_round=0,  mode='test', result_dict={"loss": [], "acc": [], "epoch": [], "eps":[],"commu_round":[],"output_norm":[]})
    else:
        server.run(val_log= conf.val_log)



if __name__ == '__main__':
    main()
    print("finish job!")
    exit(0)