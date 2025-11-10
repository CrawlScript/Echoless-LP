# coding=utf-8

import torch
import dgl

import numpy as np
from echoless_lp.utils.random_project_utils import *
from itertools import chain
import logging
from echoless_lp.global_configuration import global_config
from tqdm import tqdm

logger = logging.getLogger()




def torch_svd(x):
    u, s, vh = torch.linalg.svd(x)
    
    print("svd: ", x.size(), u.size(), s.size(), vh.size())
    h = u[:, :s.size(0)] * torch.sqrt(s)
    return h



def get_raw_etype(etype):
    etype_ = etype[1]
    if etype_.startswith("r."):
        return get_reversed_etype(etype)
    else:
        return etype

def rphgnn_propagate_then_update(g, current_k, inner_k, input_x_dim_dict, target_node_type, squash_strategy, norm=None, squash_even_odd="all", squash_self=True, collect_even_odd="all", diag_dict=None, train_label_feat=None):

    with g.local_scope():

        # propagate 
        for etype in g.canonical_etypes:
            last_key = "feat"
            for k_ in range(1, inner_k + 1):
                # print("etype: ", etype, "inner_k_: ", k_)
                odd_or_even = "odd" if k_ % 2 == 1 else "even"
                key = (odd_or_even, k_, etype)
                prop_etype = etype if odd_or_even == "odd" else get_reversed_etype(etype)

                # print("prop_etype: ", prop_etype)

                if norm == "mean":

                    g.update_all(
                        dgl.function.copy_u(last_key, "m"),
                        dgl.function.mean("m", key),

                        # message_func,
                        # dgl.function.sum("m", key),
                        
                        etype=prop_etype
                    )
                    last_key = key

                else:
                    sp = torch.tensor(norm[2])
                    dp = torch.tensor(norm[4])
                
                    def message_func(edges):
                        # return {'m': edges.src[last_key]}
                        return {'m': edges.src[last_key] * \
                                torch.pow(edges.src[("deg", get_reversed_etype(prop_etype))].unsqueeze(-1) + 1e-8, sp) * \
                                torch.pow(edges.dst[("deg", prop_etype)].unsqueeze(-1) + 1e-8, dp)}
                    
                    g.update_all(
                        message_func,
                        dgl.function.sum("m", key),
                        etype=prop_etype
                    )
                    last_key = key
                    

        new_x_dict = {}


        for ntype in g.ntypes:
            # print("deal with {} ...".format(ntype))
            
            # sort keys by (etype, k)
            # [(odd, 1, etype0), (even, 2, etype0), (odd, 3, etype0), (even, 4, etype0), 
            # (odd, 1, etype1), (even, 2, etype1), (odd, 3, etype1), (even, 4, etype1)]
            keys = [key for key in g.nodes[ntype].data.keys() 
                    if isinstance(key, tuple) and key[0] in ["even", "odd"]]
            sort_index = sorted(list(range(len(keys))), key=lambda i: (get_raw_etype(keys[i][-1]), keys[i][1]))
            sorted_keys = [keys[i] for i in sort_index]

        
            x = g.ndata["feat"][ntype]

            # collect for each ntype
            h_list = []
            
            for key in sorted_keys:
                # print(key, g.nodes[ntype].data[key].size())
                h = g.nodes[ntype].data[key]

                # label prop for target node type
                if ntype == target_node_type and diag_dict is not None:

                    if key[0] == "even":
                        diag = diag_dict[key[-1]]
                        # diag = np.expand_dims(diag, axis=-1)
                        h = (h - x * diag) / (1.0 - diag + 1e-8)

                        if train_label_feat is not None:
                            zero_mask = (h.sum(dim=-1) == 0.0)
                            h[zero_mask] = torch.ones_like(h[zero_mask]) / h.size(-1) 
                            print("diag zero to mean for: {} {} {}".format(ntype, key, zero_mask.sum()))

                        print("diag====", key)
                        print("remove diag for: {} {}".format(ntype, key))


                h_list.append(h)


        

            # each even_odd_iter covers an odd and an even, such as (1,2) or (3, 4)
            def get_even_odd_iter(data_list, i):
                """
                input: [(odd, 1, etype0), (even, 2, etype0), (odd, 3, etype0), (even, 4, etype0), (odd, 1, etype1), (even, 2, etype1), (odd, 3, etype1), (even, 4, etype1)]
                
                output: odd+even of a given iteration i
                
                For exampe, if i == 0:
                output => [(odd, 1, etype0), (even, 2, etype0), (odd, 1, etype1), (even, 2, etype1)]
                """
                return list(chain(*list(zip(data_list[i * 2::inner_k], data_list[i * 2 + 1::inner_k]))))

            even_odd_iter_h_list_list = []

            for hop in range(inner_k // 2):
                even_odd_iter_h_list = get_even_odd_iter(h_list, hop)
                even_odd_iter_h_list_list.append(even_odd_iter_h_list)
                even_odd_iter_sorted_keys = get_even_odd_iter(sorted_keys, hop)
                # print("hop sorted keys: ", hop_sorted_keys)

            
            even_odd_iter_sorted_keys = [(key[0], key[2]) for key in even_odd_iter_sorted_keys]

            # push into outputs
            if ntype == target_node_type:
                # print("collect outputs for {}".format(ntype))

                # target_h_list_list = [[h.detach().cpu().numpy() for h in hop_h_list] 
                #                         for hop_h_list in even_odd_iter_h_list_list]
                
                # target_h_list_list = [[h.detach().cpu() for h in hop_h_list] 
                #         for hop_h_list in even_odd_iter_h_list_list]
                
                target_h_list_list = [[h.detach().cpu().to(torch.float16) for h in hop_h_list] 
                        for hop_h_list in even_odd_iter_h_list_list]
                
                target_sorted_keys = even_odd_iter_sorted_keys

                if collect_even_odd != "all":
                    target_h_list_list = [[target_h for target_h, key in zip(target_h_list, target_sorted_keys) if key[0] == collect_even_odd] 
                                          for target_h_list in target_h_list_list]
                    target_sorted_keys = [key for key in target_sorted_keys if key[0] == collect_even_odd]
                    

            

            squash_keys = [("self", )] if squash_self else []
            squash_h_list = [x] if squash_self else []

            for h, key in zip(even_odd_iter_h_list_list[0], even_odd_iter_sorted_keys):
                key_even_odd = key[0]

                use_key = None
                if squash_even_odd == "all":
                    use_key = True
                elif squash_even_odd in ["even", "odd"]:
                    use_key = key_even_odd == squash_even_odd
                else:
                    raise ValueError("squash_even_odd must be all, even or odd")
                
                if use_key:
                    squash_keys.append(key)
                    squash_h_list.append(h)

            if squash_strategy == "sum":
                new_x = torch.stack(squash_h_list, dim=0).sum(dim=0)

            elif squash_strategy == "mean":
                new_x = torch.stack(squash_h_list, dim=0).mean(dim=0)

            elif squash_strategy == "sum_norm":
                new_x = torch_normalize(torch.stack(squash_h_list, dim=0).sum(dim=0))

            elif squash_strategy == "norm_sum":
                normed_squash_h_list = [torch_normalize(h) for h in squash_h_list]
                new_x = torch.stack(normed_squash_h_list, dim=0).sum(dim=0)

            elif squash_strategy == "norm_mean":
                normed_squash_h_list = [torch_normalize(h) for h in squash_h_list]
                new_x = torch.stack(normed_squash_h_list, dim=0).mean(dim=0)

            elif squash_strategy == "norm_mean_norm":
                normed_squash_h_list = [torch_normalize(h) for h in squash_h_list]
                h = torch.stack(normed_squash_h_list, dim=0).mean(dim=0)
                h = torch_normalize(h)
                new_x = h

            elif squash_strategy == "project_norm_sum":
                new_x = torch_random_project_then_sum(
                    squash_h_list,
                    input_x_dim_dict[ntype],
                    norm=True
                )
            
            elif squash_strategy == "project_norm_mean":
                new_x = torch_random_project_then_mean(
                    squash_h_list,
                    input_x_dim_dict[ntype],
                    norm=True
                )
            else:
                raise ValueError("wrong squash_strategy: {}".format(squash_strategy))

            new_x_dict[ntype] = new_x

    # print("update ndata")
    for ntype, new_x in new_x_dict.items():
        g.nodes[ntype].data["feat"] = new_x

    if target_node_type is None:
        target_sorted_keys = None
        target_h_list_list = None

    return (target_h_list_list, target_sorted_keys), g


def compute_deg_dict(g):

    with torch.no_grad():

        deg_dict = {}
        def message_func(edges):
            return {'m': torch.ones([len(edges)])}

        for etype in g.canonical_etypes:
            key = ("deg", etype)
            g.update_all(
                message_func,
                dgl.function.sum("m", key),
                etype=etype
            )
            deg = g.ndata[key][etype[-1]]
            deg_dict[etype] = deg

    return deg_dict


def compute_diag_dict(g):
    import scipy.sparse as sp
    import numpy as np

    diag_dict = {}

    def norm_adj(adj):
        deg = np.array(adj.sum(axis=-1)).flatten()
        inv_deg = 1.0 / deg
        inv_deg[np.isnan(inv_deg)] = 0.0
        inv_deg[np.isinf(inv_deg)] = 0.0

        normed_adj = sp.diags(inv_deg) @ adj
        return normed_adj

    with torch.no_grad():

        for etype in g.canonical_etypes:
            src, dst = g.edges(etype=etype)
            src = src.detach().cpu().numpy()
            dst = dst.detach().cpu().numpy()

            shape = [g.num_nodes(etype[0]), g.num_nodes(etype[-1])]

            adj = sp.csr_matrix((np.ones_like(src), (src, dst)), shape=shape)

            
            diag = (norm_adj(adj).multiply(norm_adj(adj.T).T)).sum(axis=-1)
            diag = np.array(diag).flatten().astype(np.float32)
        
            diag = np.expand_dims(diag, axis=-1)
            diag_dict[etype] = torch.tensor(diag)

            # print("compute diag for {}: {}".format(etype, diag))


    return diag_dict




def rphgnn_propagate_and_collect(g, k, inner_k, alpha, target_node_type, use_input_features, squash_strategy, train_label_feat, norm, squash_even_odd, collect_even_odd, squash_self=False, target_feat_random_project_size=None, add_self_group=False, reset_train=False, train_mask=None):

    with torch.no_grad():

        raw_input_target_x = g.ndata["feat"][target_node_type]

        with g.local_scope():

            featureless_node_types = [ntype for ntype in g.ntypes if ntype != target_node_type]
            embedding_size = g.ndata["feat"][featureless_node_types[0]].size(-1)

            if target_feat_random_project_size is not None:
                new_x = global_config.torch_random_project(raw_input_target_x, target_feat_random_project_size, norm=True)
                g.nodes[target_node_type].data["feat"] = new_x
                print("random_project_target_feat {} => {}...".format(raw_input_target_x.size(-1), new_x.size(-1)))

            if train_label_feat is not None:

                num_classes = train_label_feat.size(-1)
                for ntype in g.ntypes:
                    if ntype == target_node_type:
                        g.nodes[ntype].data["feat"] = train_label_feat
                    else:
                        g.nodes[ntype].data["feat"] = torch.ones([g.num_nodes(ntype), num_classes]) / num_classes

                diag_dict = compute_diag_dict(g)

            else:
                diag_dict = None

            input_x_dim_dict = {
                ntype: g.ndata["feat"][ntype].size(-1)
                for ntype in g.ntypes
            }

            input_x_dict = {
                ntype: g.ndata["feat"][ntype] for ntype in g.ntypes
            }

                

            input_target_x = g.ndata["feat"][target_node_type]#.detach().cpu().numpy()
            target_h_list_list = []
            for k_ in range(k):
                # print("start propagate {} ...".format(k_))
                
                # print("start {} propagate-then-update iteration {} ...".format("feat" if train_label_feat is None else "pre-label", k_))
                print("start propagate-then-update iteration {} ...".format(k_))
                (target_h_list_list_, target_sorted_keys), g = rphgnn_propagate_then_update(g, k_, inner_k, input_x_dim_dict, target_node_type, squash_strategy=squash_strategy, norm=norm, squash_even_odd=squash_even_odd, collect_even_odd=collect_even_odd, squash_self=squash_self, diag_dict=diag_dict, train_label_feat=train_label_feat)
               
                target_h_list_list.extend(target_h_list_list_)


                for ntype in g.ntypes:
                    g.nodes[ntype].data["feat"] = g.nodes[ntype].data["feat"] * (1 - alpha) + input_x_dict[ntype] * alpha

                if reset_train:
                    g.nodes[target_node_type].data["feat"] = torch.where(
                        train_mask.unsqueeze(-1),
                        raw_input_target_x,
                        g.nodes[target_node_type].data["feat"]
                    )

                    print("==== reset train part after k-th iteration: ", k_)


            target_h_list_list = [list(target_h_list) for target_h_list in zip(*target_h_list_list)]


        target_sorted_keys_ = target_sorted_keys[:]
        target_h_list_list_ = target_h_list_list[:]


        if train_label_feat is not None:

            target_sorted_keys = []
            target_h_list_list = []
            for key, target_h_list in zip(target_sorted_keys_, target_h_list_list_):
                if key[0] == "even":
                    target_sorted_keys.append(key)
                    target_h_list_list.append(target_h_list)
                elif key[0] == "odd":
                    etype = key[-1]
                    if etype[0] == etype[-1]:
                        print("add homo for label: ", key)
                        target_sorted_keys.append(key)
                        target_h_list_list.append(target_h_list)

            target_h_list_list = [target_h_list[-1:] for target_h_list in target_h_list_list]

            
        if use_input_features:
            for target_h_list, key in zip(target_h_list_list, target_sorted_keys):
                if key[0] in ["even", "self"]:

                    print("add input x to {}".format(key))
                    x = input_target_x
                    # x = x.detach().cpu().numpy()
                    x = x.detach().cpu()

                    target_h_list.insert(0, x)


        # for target_h_list in target_h_list_list:
        #     print("context: ")
        #     for target_h in target_h_list:
        #         print(target_h.shape)

    if add_self_group:
        # target_h_list_list.append([raw_input_target_x.detach().cpu().numpy()])
        target_h_list_list.append([raw_input_target_x.detach().cpu()])
        target_sorted_keys.append(("self",))

    print("target_sorted_keys: ", target_sorted_keys)
    # target_h_list_list = [np.stack(target_h_list, axis=1) for target_h_list in target_h_list_list]


    target_h_list_list = [torch.stack([target_h.to(torch.float16) for target_h in target_h_list], dim=1)
                          for target_h_list in target_h_list_list]


    return target_h_list_list, target_sorted_keys



def rphgnn_propagate_and_collect_label(hetero_graph, target_node_type, y, train_label_feat):

    label_target_h_list_list, _ = rphgnn_propagate_and_collect(hetero_graph, 
                1, 
                2, 
                0.0,
                target_node_type, use_input_features=False, 
                squash_strategy="mean", 
                train_label_feat=train_label_feat, 
                norm="mean",  
                squash_even_odd="all",
                collect_even_odd="all"
                )  
    

    return label_target_h_list_list




def rphgnn_echoless_propagate_and_collect_label(hetero_graph, target_node_type, y, train_label_feat, 
                                                label_k,
                                                num_partitions=None, 
                                                extra_mask=None,
                                                train_mask=None,
                                                lp_squash_strategy=None,
                                                renorm=False,
                                                reset_train=False,
                                                label_mask_rate=0.0
                                                ):
    
    if label_k == 0:
        return []

    num_target_nodes = hetero_graph.num_nodes(target_node_type)

    num_classes = train_label_feat.size(-1)

    if renorm:
        train_label_feat = torch.concat([
            train_mask.float().unsqueeze(-1),
            train_label_feat
        ], dim=-1)
    num_label_feats = train_label_feat.size(-1)

    labels = torch.tensor(y)


    def create_partition_uniform():
        if num_partitions is None or num_partitions == 0:
            partition = None
        else:
            num_repeat = np.ceil(hetero_graph.num_nodes(target_node_type) / num_partitions).astype(int)
            partition = torch.arange(num_partitions).tile(num_repeat)[:hetero_graph.num_nodes(target_node_type)]
            partition = partition[torch.randperm(hetero_graph.num_nodes(target_node_type))]
        return partition
            

    def create_uniform_partition(num_nodes, num_partitions):
        num_repeat = np.ceil(num_nodes / num_partitions).astype(int)
        partition = torch.arange(num_partitions).tile(num_repeat)[:num_nodes]
        partition = partition[torch.randperm(num_nodes)]
        return partition


    def create_partition_class_uniform():

        if num_partitions is None or num_partitions == 0:
            partition = None
        else:
            partition = torch.ones(hetero_graph.num_nodes(target_node_type), dtype=torch.long) * -1

            train_class_masks = [train_mask & (labels == i) for i in range(num_classes)]
            valid_test_mask = ~train_mask

            masks = train_class_masks + [valid_test_mask]

            print("generate partition for each class and valid/test set")
            for mask in tqdm(masks):
                partition[mask] = create_uniform_partition(mask.sum().item(), num_partitions)

        return partition
    



            

    def create_partition():
        if False and labels.dim() == 1: # and args.dataset not in ["oag_venue", "oag_L1"]:
            partition = create_partition_class_uniform()
        else:
            partition = create_partition_uniform()
        return partition




    # if num_partitions is None or num_partitions == 0:
    #     partition = None
    # else:
    #     num_repeat = np.ceil(num_target_nodes / num_partitions).astype(int)
    #     partition = torch.arange(num_partitions).tile(num_repeat)[:num_target_nodes]
    #     partition = partition[torch.randperm(num_target_nodes)]

    
    partition = create_partition()


    # num_label_k = 4 


    with hetero_graph.local_scope():
        for ntype in hetero_graph.ntypes:
            if ntype != target_node_type:
                hetero_graph.nodes[ntype].data["feat"] = torch.zeros([hetero_graph.num_nodes(ntype), num_label_feats])

        if partition is None:


            hetero_graph.nodes[target_node_type].data["feat"] = train_label_feat
            output_label_target_h_list_list, _ = rphgnn_propagate_and_collect(
                hetero_graph, 
                label_k,                                             
                2, 
                0.0,
                target_node_type, use_input_features=False, 
                # squash_strategy="sum", 
                squash_strategy="mean", 
                train_label_feat=None, 
                norm="mean",
                # squash_even_odd="all",
                # collect_even_odd="all",
                squash_even_odd="odd",
                collect_even_odd="odd"
            )  
        else:
            

            def masked_prop(mask, scale, output_label_target_h_list_list, label_mask_rate=0.0):

                masked_train_label_feat = torch.where(
                    mask.unsqueeze(-1), 
                    torch.zeros_like(train_label_feat), 
                    train_label_feat
                ) * scale

                if label_mask_rate is not None and label_mask_rate > 0:
                    print("apply label mask with rate: ", label_mask_rate)
                    label_mask = torch.rand(mask.size()) <= label_mask_rate
                    masked_train_label_feat = torch.where(label_mask.unsqueeze(-1), 0.0, masked_train_label_feat)
                    masked_train_label_feat = masked_train_label_feat / (1 - label_mask_rate)



                hetero_graph.nodes[target_node_type].data["feat"] = masked_train_label_feat

                label_target_h_list_list, _ = rphgnn_propagate_and_collect(hetero_graph, 
                            label_k,                                           
                            2, 
                            0.0,
                            target_node_type, use_input_features=False, 
                            # squash_strategy="sum", 
                            # squash_strategy="mean", 
                            squash_strategy=lp_squash_strategy,


                            # squash_strategy="norm_sum",

                            # squash_strategy="norm_mean",
                            # squash_strategy="norm_sum",

                            # squash_strategy="sum_norm",


                            # squash_strategy="norm_mean_norm",
                            
                            train_label_feat=None, 
                            norm="mean",  
                            squash_even_odd="all",
                            collect_even_odd="all",
                            # squash_even_odd="odd",
                            # collect_even_odd="odd",
                            reset_train=reset_train,
                            train_mask=train_mask
                            )  
            
                if output_label_target_h_list_list is None:
                    # output_label_target_h_list_list = [np.zeros_like(label_target_h_list) for label_target_h_list in label_target_h_list_list]
                    output_label_target_h_list_list = [torch.zeros_like(label_target_h_list) for label_target_h_list in label_target_h_list_list]


                for i, label_target_h_list in enumerate(label_target_h_list_list):
                    output_label_target_h_list_list[i][mask] = label_target_h_list[mask]

                return output_label_target_h_list_list


            output_label_target_h_list_list = None

            for partition_index in range(num_partitions):
                print("start partition: ", partition_index)
                mask = partition == partition_index


                if renorm:
                    scale = num_partitions / (num_partitions - 1)
                else:
                    scale = 1.0

                output_label_target_h_list_list = masked_prop(mask, scale, 
                                                            output_label_target_h_list_list,
                                                            label_mask_rate=label_mask_rate)

            if extra_mask is not None:
                print("start extra partition: ", partition_index)
                mask = extra_mask
                scale = 1.0
                output_label_target_h_list_list = masked_prop(mask, scale, 
                                                              output_label_target_h_list_list,
                                                              label_mask_rate=0.0)

    if renorm:
        def renorm_label_h(label_h):
            
            # import pdb
            # pdb.set_trace()

            convert_type = True
            if convert_type:
                input_dtype = label_h.dtype
                label_h = label_h.to(torch.float32)

            train_score = label_h[:, 0]
            label_h = label_h[:, 1:]




            mean_train_score = train_score.mean()
            scale = mean_train_score / (train_score + 1e-8)
            # scale = torch.sqrt(scale.to(torch.float32)).to(scale.dtype)

            renormed_label_h = label_h * scale.unsqueeze(-1)

            zero_train_mask = train_score == 0.0

            renormed_label_h[zero_train_mask] = label_h[zero_train_mask].clone()

            if convert_type:
                renormed_label_h = renormed_label_h.to(input_dtype)

            return renormed_label_h
        
        def renorm_label_h_list(label_h_list):
            return torch.stack([renorm_label_h(label_h_list[:, i]) for i in range(label_h_list.size(1))], dim=1)
        
        
        output_label_target_h_list_list = [renorm_label_h_list(label_target_h_list) for label_target_h_list in output_label_target_h_list_list]



    print("before masking zero")
    for h in output_label_target_h_list_list:
        print(h.shape)
    
    if False:
        non_zero_masks = [np.abs(label_target_h_list).mean(axis=0).mean(axis=-1) != 0.0 for label_target_h_list in output_label_target_h_list_list]
        output_label_target_h_list_list = [label_target_h_list[:, non_zero_mask] for label_target_h_list, non_zero_mask in zip(output_label_target_h_list_list, non_zero_masks)]

    print("after masking zero")
    for h in output_label_target_h_list_list:
        print(h.shape)

    transpose = False
    # transpose = True

    if transpose:
        print("before transpose")
        for h in output_label_target_h_list_list:
            print(h.shape)

        output_label_target_h_list_list_ = output_label_target_h_list_list
        output_label_target_h_list_list = []

        for i in range(output_label_target_h_list_list_[0].shape[1]):
            # output_label_target_h_list_list.append(np.stack([h[:, i] for h in output_label_target_h_list_list_], axis=1))
            
            output_label_target_h_list_list.append(torch.stack([h[:, i] for h in output_label_target_h_list_list_], dim=1))

        print("after transpose")
        for h in output_label_target_h_list_list:
            print(h.shape)



    

    # for h in output_label_target_h_list_list:
    #     print(h.shape)
    # asdfadf

    # for m in zero_masks:
    #     print(m)
    # asdfasdf
    

    # for i, h in enumerate(output_label_target_h_list_list):
    #     for j in range(h.shape[1]):
    #         sub_h = h[:, j]
    #         sum_v = np.abs(sub_h).mean()
    #         print(i, j, sum_v, sub_h.shape)
    # asdfsad

    flatten = False

    if flatten:
        output_label_target_h_list_list_ = output_label_target_h_list_list
        output_label_target_h_list_list = []

        for i in range(output_label_target_h_list_list_[0].shape[1]):
            output_label_target_h_list_list.extend([h[:, i:i+1] for h in output_label_target_h_list_list_])


        
    
    # for i, h in enumerate(output_label_target_h_list_list):
    #     print(i, h.shape)

    # asdfasdf

    return output_label_target_h_list_list









def multi_rphgnn_echoless_propagate_and_collect_label(hetero_graph, target_node_type, y, train_label_feat, 
                                                label_k,
                                                num_partitions=None, 
                                                extra_mask=None,
                                                train_mask=None,
                                                num_lp_repeats=1,
                                                lp_squash_strategy=None,
                                                renorm=False,
                                                reset_train=False,
                                                label_mask_rate=0.0
                                                ):
    
    label_target_h_list_list_sum = None

    for i in range(num_lp_repeats):
        label_target_h_list_list = rphgnn_echoless_propagate_and_collect_label(hetero_graph, target_node_type, y, train_label_feat, 
                                                label_k,
                                                num_partitions, 
                                                extra_mask,
                                                train_mask,
                                                lp_squash_strategy=lp_squash_strategy,
                                                renorm=renorm,
                                                reset_train=reset_train,
                                                label_mask_rate=label_mask_rate
                                                )
        
        if num_lp_repeats == 1:
            return label_target_h_list_list
        
        if label_target_h_list_list_sum is None:
            label_target_h_list_list_sum = label_target_h_list_list
        else:
            for i, label_target_h_list in enumerate(label_target_h_list_list):
                label_target_h_list_list_sum[i] += label_target_h_list


    label_target_h_list_list = [label_target_h_list / num_lp_repeats for label_target_h_list in label_target_h_list_list_sum]

    return label_target_h_list_list