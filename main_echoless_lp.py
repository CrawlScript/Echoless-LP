

from itertools import chain
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics

import shutil
import logging
import sys
import time
import os
import json
import datetime
import shortuuid
from argparse import ArgumentParser
from echoless_lp.utils.argparse_utils import parse_bool



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


use_wandb = False

parser = ArgumentParser()

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--method", type=str, required=True)
parser.add_argument("--use_nrl", type=parse_bool, required=True)
parser.add_argument("--use_input", type=parse_bool, required=True)
parser.add_argument("--use_label", type=parse_bool, required=True)
parser.add_argument("--even_odd", type=str, required=False, default="all")
parser.add_argument("--use_all_feat", type=parse_bool, required=True)
parser.add_argument("--train_strategy", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--gpus", type=str, required=True)
parser.add_argument("--input_drop_rate", type=float, required=False, default=None)
parser.add_argument("--drop_rate", type=float, required=False, default=None)
parser.add_argument("--label_input_drop_rate", type=float, required=False, default=None)

parser.add_argument("--label_emb_size", type=int, required=False, default=None)

parser.add_argument("--hidden_size", type=int, required=False, default=None)
parser.add_argument("--squash_k", type=int, required=False, default=None)
parser.add_argument("--num_partitions", type=int, required=False, default=None)
parser.add_argument("--use_extra_mask", type=parse_bool, required=True)


parser.add_argument("--use_renorm", type=parse_bool, required=True)

parser.add_argument("--label_k", type=int, required=False, default=None)
parser.add_argument("--num_epochs", type=int, required=False, default=None)
parser.add_argument("--max_patience", type=int, required=False, default=None)
parser.add_argument("--embedding_size", type=int, required=False, default=None)


parser.add_argument("--label_mask_rate", type=float, required=False, default=0.0)

parser.add_argument("--rps", type=str, required=False, default="sp_3.0", help="random projection strategies")
parser.add_argument("--seed", type=int, required=True)




parser.add_argument("--feat_mode", type=str, required=True)


parser.add_argument("--label_merge_mode", type=str, required=True)
parser.add_argument("--num_lp_repeats", type=int, required=True)


parser.add_argument("--lp_squash_strategy", type=str, required=False, default="mean")




args = parser.parse_args()

method = args.method 
dataset = args.dataset 
use_all_feat = args.use_all_feat
use_nrl = args.use_nrl
use_label = args.use_label
train_strategy = args.train_strategy
use_input_features = args.use_input
output_dir = args.output_dir
gpu_ids = args.gpus
device = "cuda"
data_loader_device = device
even_odd = args.even_odd
random_projection_strategy = args.rps
seed = args.seed

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids



from echoless_lp.callbacks import CSVLoggingCallback, EarlyStoppingCallback, LoggingCallback, TensorBoardCallback
from echoless_lp.layers.rphgnn_encoder import RpHGNNEncoder
from echoless_lp.layers.rphgnn_gt_encoder import RpHGNNGTEncoder
from echoless_lp.losses import kl_loss
from echoless_lp.utils.metrics_utils import MRR, NDCG
from echoless_lp.utils.random_project_utils import create_func_torch_random_project_create_kernel_sparse, torch_random_project_common, torch_random_project_create_kernel_xavier, torch_random_project_create_kernel_xavier_no_norm
from echoless_lp.utils.torch_data_utils import NestedDataLoader
from echoless_lp.global_configuration import global_config

from echoless_lp.utils.random_utils import reset_seed
from echoless_lp.configs.default_param_config import load_default_param_config
from echoless_lp.datasets.load_data import  load_dgl_data
from echoless_lp.utils.nested_data_utils import gather_h_y, nested_gather, nested_map
from echoless_lp.layers.rphgnn_pre import multi_rphgnn_echoless_propagate_and_collect_label, rphgnn_propagate_and_collect, rphgnn_propagate_and_collect_label, rphgnn_echoless_propagate_and_collect_label


np.set_printoptions(precision=4, suppress=True)




reset_seed(seed)
print("seed = ", seed)





global_config.torch_random_project = torch_random_project_common
if random_projection_strategy.startswith("sp"):
    random_projection_sparsity = float(random_projection_strategy.split("_")[1])
    global_config.torch_random_project_create_kernel = create_func_torch_random_project_create_kernel_sparse(s=random_projection_sparsity)
    print("setting random projection strategy: sparse({} ...)".format(random_projection_sparsity))
elif random_projection_strategy == "gaussian":
    global_config.torch_random_project_create_kernel = torch_random_project_create_kernel_xavier
    print("setting random projection strategy: gaussian ...")

elif random_projection_strategy == "gaussian_no_norm":
    global_config.torch_random_project_create_kernel = torch_random_project_create_kernel_xavier_no_norm
    print("setting random projection strategy: gaussian ...")

else:
    raise ValueError("unknown random projection strategy: {}".format(random_projection_strategy))


pre_device = "cpu"
learning_rate = 3e-3
l2_coef = None
norm = "mean"
squash_strategy = "project_norm_sum"
target_h_dtype = torch.float16


timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

running_leaderboard_mag = dataset == "mag" and train_strategy == "cl"  and use_label

if running_leaderboard_mag:
    scheduler_gamma = 0.99

    num_views = 3
    cl_rate = 0.6

    model_save_dir = "saved_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, "leaderboard_mag_seed_{}.pt".format(seed))
else:
    scheduler_gamma = None
    model_save_path = None
    if train_strategy == "common":
        num_views = 1
        cl_rate = None
    else:
        num_views = 2
        cl_rate = 0.5






arg_dict = {**vars(args)}
arg_dict["date"] = timestamp
del arg_dict["output_dir"]
del arg_dict["gpus"]

args_desc_items = []
for key, value in arg_dict.items():
    args_desc_items.append(key)
    args_desc_items.append(str(value))
args_desc = "_".join(args_desc_items)

uuid = "{}_{}".format(timestamp, shortuuid.uuid())

tmp_output_fname = "{}.json.tmp".format(uuid)
tmp_output_fpath = os.path.join(output_dir, tmp_output_fname)

output_fname = "{}.json".format(uuid)
output_fpath = os.path.join(output_dir, output_fname)


print(output_dir)
print(os.path.exists(output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(tmp_output_fpath, "a", encoding="utf-8") as f:
    f.write("{}\n".format(json.dumps(arg_dict)))


if use_wandb:

    import wandb
    wandb.init(
        config=arg_dict
    )


time_dict = {
    "start": time.time()
}

squash_k, inner_k, conv_filters, num_layers_list, hidden_size, merge_mode, input_drop_rate, drop_rate, \
        use_pretrain_features, random_projection_align, input_random_projection_units, target_feat_random_project_size, add_self_group = load_default_param_config(dataset)


embedding_size = None

if args.embedding_size is not None:
    embedding_size = args.embedding_size
    print("reset embedding_size => {}".format(embedding_size))

with torch.no_grad():
    hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index), \
            batch_size, num_epochs, patience, validation_freq, convert_to_tensor = load_dgl_data(
        dataset,
        use_all_feat=use_all_feat,
        embedding_size=embedding_size,
        use_nrl=use_nrl
    )

for ntype in hetero_graph.ntypes:
    print(ntype, hetero_graph.number_of_nodes(ntype), hetero_graph.nodes[ntype].data["feat"].size())

for etype in hetero_graph.canonical_etypes:
    print(etype, hetero_graph.number_of_edges(etype))




if args.input_drop_rate is not None:
    input_drop_rate = args.input_drop_rate
    print("reset input_drop_rate => {}".format(input_drop_rate))
    
if args.drop_rate is not None:
    drop_rate = args.drop_rate
    print("reset drop_rate => {}".format(drop_rate))
    
if args.hidden_size is not None:
    hidden_size = args.hidden_size
    print("reset hidden_size => {}".format(hidden_size))

if args.squash_k is not None:
    squash_k = args.squash_k
    print("reset squash_k => {}".format(squash_k))

if args.num_epochs is not None:
    num_epochs = args.num_epochs
    print("reset num_epochs => {}".format(num_epochs))

if args.max_patience is not None:
    patience = args.max_patience
    print("reset patience => {}".format(patience))


label_input_drop_rate = args.label_input_drop_rate

y = hetero_graph.ndata["label"][target_node_type].detach().cpu().numpy()

print("train_rate = {}\tvalid_rate = {}\ttest_rate = {}".format(len(train_index) / len(y), len(valid_index) / len(y), len(test_index) / len(y)))

multi_label = len(y.shape) > 1

if multi_label:
    num_classes = y.shape[-1]
else:
    num_classes = y.max() + 1



stage_output_dict = {
    "last": None
}


print("start pre-computation ...")

log_dir = "logs/{}".format(args_desc)

torch_y = torch.tensor(y).long()

if multi_label:
    torch_y = torch_y.float()

train_mask = np.zeros([len(y)])
train_mask[train_index] = 1.0
torch_train_mask = torch.tensor(train_mask).bool()

if even_odd == "odd":
    squash_k *= 2
    print("odd mode, squash_k =", squash_k)





label_merge_mode = args.label_merge_mode








if args.use_extra_mask:
    extra_mask = torch.zeros([hetero_graph.num_nodes(target_node_type)], dtype=torch.bool)
    
    extra_mask[test_index] = True
else:
    extra_mask = None

def create_label_target_h_list_list():
    print("using new train_label_feat")
    

    train_label_feat = torch.zeros([len(y), num_classes]).float() 

    if multi_label:
        train_label_feat[train_index] = torch.tensor(y[train_index]).float()
    else:
        train_label_feat[train_index] = F.one_hot(torch.tensor(y[train_index]), num_classes).float()

    if args.label_emb_size is not None and args.label_emb_size > 0:
        rand_weight = torch.randn(num_classes, args.label_emb_size) / np.sqrt(args.label_emb_size)
        train_label_feat = train_label_feat @ rand_weight
        print("project label feat from {} to {}".format(num_classes, args.label_emb_size))


 
    


    label_target_h_list_list = multi_rphgnn_echoless_propagate_and_collect_label(hetero_graph, 
                                                                        target_node_type, 
                                                                        y, 
                                                                        train_label_feat,
                                                                        label_k=args.label_k,
                                                                        num_partitions=args.num_partitions,
                                                                        extra_mask=extra_mask,
                                                                        train_mask=torch_train_mask,
                                                                        num_lp_repeats=args.num_lp_repeats,
                                                                        lp_squash_strategy=args.lp_squash_strategy,
                                                                        renorm=args.use_renorm,
                                                                        
                                                                        reset_train=False,
                                                                        label_mask_rate=args.label_mask_rate
                                                                        )

   
    
    
    label_target_h_list_list = nested_map(label_target_h_list_list, lambda x: x.to(target_h_dtype).to(pre_device))


    print("label_target_h_list_list")

    for i, h in enumerate(label_target_h_list_list):
        print(i, h.size())

    if use_label and not multi_label:
        for i, label_target_h_list in enumerate(label_target_h_list_list):
            for j in range(label_target_h_list.size(1)):
                label_target_h = label_target_h_list[:, j].float()

                # print(label_target_h)
                

                # print("===== group = {}\thop ={}".format(i, j))
                y_preds = label_target_h.argmax(dim=-1)
                train_acc = (y_preds[train_index] == torch_y[train_index]).float().mean().item()
                valid_acc = (y_preds[valid_index] == torch_y[valid_index]).float().mean().item()
                test_acc = (y_preds[test_index] == torch_y[test_index]).float().mean().item()
                # print("train_acc = {}".format(train_acc))
                # print("valid_acc = {}".format(valid_acc))
                # print("test_acc = {}".format(test_acc))


    

    if label_merge_mode == "concat":
        print("merge labels: ", label_merge_mode)
        combined_label_target_h_list = torch.cat(label_target_h_list_list, dim=-1)
        label_target_h_list_list = [combined_label_target_h_list]

    if label_merge_mode == "last":
        print("merge labels: ", label_merge_mode)
        label_target_h_list_list = [label_target_h_list[:, -1:] for label_target_h_list in label_target_h_list_list]

    if label_merge_mode == "flatten":

        def flatten(h):
            return torch.split(h, 1, dim=1)
        label_target_h_list_list = list(chain(*[flatten(h) for h in label_target_h_list_list]))

    elif label_merge_mode == "concat_mean":
        print("merge labels: ", label_merge_mode)
        combined_label_target_h_list = torch.cat(label_target_h_list_list, dim=-1).mean(dim=1, keepdim=True)
        label_target_h_list_list = [combined_label_target_h_list]

    elif label_merge_mode == "global_mean":
        print("merge labels: ", label_merge_mode)
        mean_label_target_h_list = [h.mean(dim=1, keepdim=True) for h in label_target_h_list_list]
        combined_label_target_h_list = torch.stack(mean_label_target_h_list, dim=1).mean(dim=1)
        label_target_h_list_list = [combined_label_target_h_list]

    elif label_merge_mode == "mean_high_append":
        print("merge labels: ", label_merge_mode)
        def merge_each(h):
            if h.size(1) <= 2:
                return h
            
            low_hop_h = h[:, :2]
            high_hop_h = h[:, 2:]
            mean_high_hop_h = high_hop_h.mean(dim=1, keepdim=True)

            h = torch.cat([low_hop_h, mean_high_hop_h], dim=1)

            return h
        
        label_target_h_list_list = [merge_each(h) for h in label_target_h_list_list]


    return label_target_h_list_list







if use_label:   
    
    
    label_target_h_list_list = create_label_target_h_list_list()
else:
    label_target_h_list_list = [] 






feat_mode = args.feat_mode

if feat_mode == "all_feat":

    feat_target_h_list_list, target_sorted_keys = rphgnn_propagate_and_collect(hetero_graph, 
                            squash_k, 
                            inner_k, 
                            0.0,
                            target_node_type, 
                            use_input_features=use_input_features, squash_strategy=squash_strategy, 
                            train_label_feat=None, 
                            norm=norm,
                            squash_even_odd=even_odd,
                            collect_even_odd=even_odd,
                            squash_self=False,
                            target_feat_random_project_size=target_feat_random_project_size,
                            add_self_group=add_self_group
                            )  

    
    feat_target_h_list_list = nested_map(feat_target_h_list_list, lambda x: x.to(target_h_dtype).to(pre_device))

elif feat_mode == "self_feat":
    feat_target_h_list_list = [hetero_graph.nodes[target_node_type].data["feat"].unsqueeze(1).to(target_h_dtype).to(pre_device)]

elif feat_mode == "no_feat":
    
    feat_target_h_list_list = []

target_h_list_list = feat_target_h_list_list + label_target_h_list_list


if dataset in ["mag"]:
    if not running_leaderboard_mag:
        target_h_list_list = [target_h_list.to("cuda") if i >= len(target_h_list_list) - 3 else target_h_list
                            for i, target_h_list in enumerate(target_h_list_list)]
    else:
        target_h_list_list = [target_h_list.to("cuda") if i >= len(target_h_list_list) - 2 else target_h_list
                    for i, target_h_list in enumerate(target_h_list_list)]

elif dataset in ["oag_L1"]:
    target_h_list_list = [target_h_list.to("cuda") if i >= len(target_h_list_list) - 12 else target_h_list
                    for i, target_h_list in enumerate(target_h_list_list)]



print("size of target_h_list_list =============")
for i, h in enumerate(target_h_list_list):
    print(i, h.size())



time_dict["pre_compute"] = time.time()
pre_compute_time = time_dict["pre_compute"] - time_dict["start"]
print("pre_compute time: ", pre_compute_time)


accuracy_metric = torchmetrics.Accuracy("multilabel", num_labels=int(num_classes)) if multi_label else torchmetrics.Accuracy("multiclass" if multi_label else "multiclass", num_classes=int(num_classes)) 
if dataset in ["oag_L1", "oag_venue"]:
    metrics_dict = {
        "accuracy": accuracy_metric,
        "ndcg": NDCG(),
        "mrr": MRR()
    }
else:
    metrics_dict = {
        "accuracy": accuracy_metric,
        "micro_f1": torchmetrics.F1Score(task="multilabel", num_labels=int(num_classes), average="micro") if multi_label else torchmetrics.F1Score(task="multiclass", num_classes=int(num_classes), average="micro"),
        "macro_f1": torchmetrics.F1Score(task="multilabel", num_labels=int(num_classes), average="macro") if multi_label else torchmetrics.F1Score(task="multiclass", num_classes=int(num_classes), average="macro"),
    }
metrics_dict = {metric_name: metric.to(device) for metric_name, metric in metrics_dict.items()}



model_name = "rphgnn"


print("create model ====")
if model_name == "rphgnn":

    model = RpHGNNEncoder(
        conv_filters, 
        [hidden_size] * num_layers_list[0],
        [hidden_size] * (num_layers_list[2] - 1) + [num_classes],
        merge_mode,
        input_shape=nested_map(target_h_list_list, lambda x: list(x.size())),
        num_label_groups=len(label_target_h_list_list),
        input_drop_rate=input_drop_rate,
        drop_rate=drop_rate,
        label_input_drop_rate=label_input_drop_rate,
        
        activation="prelu",
        output_activation=None,

        metrics_dict=metrics_dict,
        multi_label=multi_label, 
        loss_func=kl_loss if dataset == "oag_L1" else None,
        learning_rate=learning_rate, 
        scheduler_gamma=scheduler_gamma,
        train_strategy=train_strategy, 
        num_views=num_views, 
        cl_rate=cl_rate

        ).to(device)

elif model_name == "rphgnn_gt":

    model = RpHGNNGTEncoder(
        conv_filters, 
        [hidden_size] * num_layers_list[0],
        [hidden_size] * (num_layers_list[2] - 1) + [num_classes],
        merge_mode,
        input_shape=nested_map(target_h_list_list, lambda x: list(x.size())),
        num_label_groups=len(label_target_h_list_list),
        input_drop_rate=input_drop_rate,
        drop_rate=drop_rate,
        label_input_drop_rate=label_input_drop_rate,
        
        activation="prelu",
        output_activation=None,

        metrics_dict=metrics_dict,
        multi_label=multi_label, 
        loss_func=kl_loss if dataset == "oag_L1" else None,
        learning_rate=learning_rate, 
        scheduler_gamma=scheduler_gamma,
        train_strategy=train_strategy, 
        num_views=num_views, 
        cl_rate=cl_rate

        ).to(device)

print(model)

print("number of params:", sum(p.numel() for p in model.parameters()))
logging_callback = LoggingCallback(tmp_output_fpath, {"pre_compute_time": pre_compute_time}, use_wandb=use_wandb)
tensor_board_callback = TensorBoardCallback(
    "logs/{}/{}".format(dataset, timestamp)
)



def train_and_eval():
      
    train_h_list_list, train_y = nested_gather([target_h_list_list, torch_y], train_index)
    valid_h_list_list, valid_y = nested_gather([target_h_list_list, torch_y], valid_index)
    test_h_list_list, test_y = nested_gather([target_h_list_list, torch_y], test_index)


    if train_strategy == "common":
        train_data_loader = NestedDataLoader(
            [train_h_list_list, train_y],
            batch_size=batch_size, shuffle=True, device=data_loader_device
        )

    elif train_strategy == "cl":

        seen_mask = torch.zeros_like(torch_y, dtype=torch.bool)
        seen_mask[train_index] = True
        seen_mask[valid_index] = True
        seen_mask[test_index] = True

        def get_seen(x):
            print("get seen ...")
            with torch.no_grad():
                return nested_map(x, lambda x: x[seen_mask])
        
        train_data_loader = NestedDataLoader(
            [get_seen(target_h_list_list), get_seen(torch_y), get_seen(torch_train_mask)],
            batch_size=batch_size, shuffle=True, device=data_loader_device
        )

    else:
        raise Exception("invalid train strategy: {}".format(train_strategy))



    valid_data_loader =NestedDataLoader(
        [valid_h_list_list, valid_y], 
        batch_size=batch_size, shuffle=False, device=data_loader_device
    )
    test_data_loader = NestedDataLoader(
        [test_h_list_list, test_y], 
        batch_size=batch_size, shuffle=False, device=data_loader_device
    )

    if dataset in ["oag_L1", "oag_venue"]:
        early_stop_strategy = "score"
        early_stop_metric_names = ["ndcg"]
    elif dataset in ["mag"]:
        early_stop_strategy = "score"
        early_stop_metric_names = ["accuracy"]
    elif dataset in ["dblp"]:
        early_stop_strategy = "loss"
        early_stop_metric_names = ["macro_f1", "micro_f1"]
    else:
        early_stop_strategy = "score"
        early_stop_metric_names = ["macro_f1", "micro_f1"]

    print("early_stop_metric_names = {}".format(early_stop_metric_names))

    early_stopping_callback = EarlyStoppingCallback(
        early_stop_strategy, early_stop_metric_names, validation_freq, patience, test_data_loader,
        model_save_path=model_save_path
    )


                                            
   

    model.fit(
        train_data=train_data_loader,
        epochs=num_epochs,
        validation_data=valid_data_loader,
        validation_freq=validation_freq,
        callbacks=[early_stopping_callback, logging_callback, tensor_board_callback],
    )



    if running_leaderboard_mag:
        from ogb.nodeproppred import Evaluator
        evaluator = Evaluator("ogbn-mag")

        print("loading saved model ...")
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        
        with torch.no_grad():
            valid_y_pred = model.predict(valid_data_loader).argmax(dim=-1, keepdim=True)
            test_y_pred = model.predict(test_data_loader).argmax(dim=-1, keepdim=True)
            ogb_valid_acc = evaluator.eval({
                'y_true': torch_y[valid_index].unsqueeze(-1),
                'y_pred': valid_y_pred
            })['acc']
            ogb_test_acc = evaluator.eval({
                'y_true': torch_y[test_index].unsqueeze(-1),
                'y_pred': test_y_pred
            })['acc']

        print("Results of OGB Evaluator: valid_acc = {}, test_acc = {}".format(ogb_valid_acc, ogb_test_acc))

train_and_eval()

shutil.move(tmp_output_fpath, output_fpath)
print("move tmp file {} => {}".format(tmp_output_fpath, output_fpath))

if use_wandb:
    wandb.finish()
