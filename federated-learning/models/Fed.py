import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__file__)


def fed_avg(w_locals, w_glob, device):
    if len(w_locals) == 0:
        return w_glob
    w_avg = {}
    for k in w_glob.keys():
        for w_local in w_locals:
            if k not in w_avg:
                w_avg[k] = torch.zeros_like(w_glob[k], device=device)
            w_avg[k] = torch.add(w_avg[k], w_local[k])
        w_avg[k] = torch.div(w_avg[k], len(w_locals))
    return w_avg


# """ node-summarized error feedback sign SGD """
def node_summarized_sign_sgd(w_locals, w_glob, server_learning_rate, device):
    if len(w_locals) == 0:
        return w_glob
    new_w_glob = {}
    server_step = {}
    for k in w_glob.keys():
        signed_w_sum = {}
        # for each key, calculate sum
        for w_local in w_locals:
            if k not in signed_w_sum:
                signed_w_sum[k] = torch.zeros_like(w_local[k], device=device)
            signed_w_sum[k] = torch.add(signed_w_sum[k], w_local[k])
        # node sign weighted aggregation
        signed_w_sum[k] = torch.div(signed_w_sum[k], len(w_locals))
        # for each key, update w_glob by multiply sign(sum) with learning rate
        server_step[k] = torch.mul(signed_w_sum[k], server_learning_rate)
        new_w_glob[k] = torch.add(w_glob[k], server_step[k])
    return new_w_glob


# """ error feedback sign SGD """
def error_feedback_sign_sgd(w_locals, w_glob, device):
    if len(w_locals) == 0:
        return w_glob
    new_w_glob = {}
    for k in w_glob.keys():
        signed_w_sum = {}
        # for each key, calculate sum
        for w_local in w_locals:
            if k not in signed_w_sum:
                signed_w_sum[k] = torch.zeros_like(w_local[k], device=device)
            signed_w_sum[k] = torch.add(signed_w_sum[k], w_local[k])
        # node sign weighted aggregation
        signed_w_sum[k] = torch.div(signed_w_sum[k], len(w_locals))
        new_w_glob[k] = torch.add(w_glob[k], signed_w_sum[k])
    return new_w_glob


# """ sign SGD with majority vote """
def sign_sgd_mv(w_locals, w_glob, server_learning_rate, device):
    if len(w_locals) == 0:
        return w_glob
    new_w_glob = {}
    server_step = {}
    for k in w_glob.keys():
        signed_w_sum = {}
        # for each key, calculate sum
        for w_local in w_locals:
            if k not in signed_w_sum:
                signed_w_sum[k] = torch.zeros_like(w_local[k], device=device)
            signed_w_sum[k] = torch.add(signed_w_sum[k], w_local[k])
        # node sign weighted aggregation
        signed_w_sum[k] = torch.sign(signed_w_sum[k])
        server_step[k] = torch.mul(signed_w_sum[k], server_learning_rate)
        new_w_glob[k] = torch.add(w_glob[k], server_step[k])
    return new_w_glob


# """ sign SGD with election coding (Deterministic), only suitable for the network with 10 nodes """
# ref: https://github.com/jysohn1108/Election-Coding/blob/main/cifar10/main.py
# encoding matrix (G) w/ r=3.8
# [[1,0,0,0,0], [0,1,1,1,0], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]]
def sign_sgd_ec(w_locals, w_glob, server_learning_rate, device):
    if len(w_locals) == 0:
        return w_glob
    coded_w_dict = {}
    for uuid in range(len(w_locals)):
        coded_w_dict[uuid] = {}
        for k in w_glob.keys():
            if uuid % 5 == 1:
                coded_w_dict[uuid][k] = torch.sum(torch.stack([w_locals[uuid][k]]), dim=0)
            elif uuid % 5 == 2:
                coded_w_dict[uuid][k] = torch.sum(torch.stack([w_locals[uuid][k], w_locals[uuid+1][k], w_locals[uuid+2][k]]), dim=0)
            elif uuid % 5 == 0:
                coded_w_dict[uuid][k] = torch.sum(torch.stack([w_locals[uuid][k], w_locals[uuid+1][k], w_locals[uuid+2][k], w_locals[uuid+3][k], w_locals[uuid+4][k]]), dim=0)
            else:
                idx = uuid - uuid % 5
                coded_w_dict[uuid][k] = torch.sum(torch.stack([w_locals[idx][k], w_locals[idx+1][k], w_locals[idx+2][k], w_locals[idx+3][k], w_locals[idx+4][k]]), dim=0)
            coded_w_dict[uuid][k] = torch.sign(coded_w_dict[uuid][k])
    new_w_glob = {}
    server_step = {}
    for k in w_glob.keys():
        signed_w_sum = {}
        # for each key, calculate sum
        for uuid in range(len(w_locals)):
            if k not in signed_w_sum:
                signed_w_sum[k] = torch.zeros_like(coded_w_dict[uuid][k], device=device)
            signed_w_sum[k] = torch.add(signed_w_sum[k], coded_w_dict[uuid][k])
        # node sign weighted aggregation
        signed_w_sum[k] = torch.sign(signed_w_sum[k])
        server_step[k] = torch.mul(signed_w_sum[k], server_learning_rate)
        new_w_glob[k] = torch.add(w_glob[k], server_step[k])
    return new_w_glob


# """ sign SGD with robust learning rate """
# ref: https://github.com/TinfoilHat0/Defending-Against-Backdoors-with-Robust-Learning-Rate
# learning threshold theta = 0.5
def sign_sgd_rlr(w_locals, w_glob, server_learning_rate, device):
    if len(w_locals) == 0:
        return w_glob
    theta = 0.5
    threshold_node_number = len(w_locals) * theta
    new_w_glob = {}
    server_step = {}
    for k in w_glob.keys():
        signed_w_sum = {}
        # for each key, calculate sum
        for w_local in w_locals:
            if k not in signed_w_sum:
                signed_w_sum[k] = torch.zeros_like(w_local[k], device=device)
            signed_w_sum[k] = torch.add(signed_w_sum[k], w_local[k])
        # robust learning rate adjustment
        rlr = torch.where(abs(signed_w_sum[k]) > threshold_node_number, signed_w_sum[k], -signed_w_sum[k])
        sign_rlr = torch.sign(rlr)
        server_step[k] = torch.mul(sign_rlr, server_learning_rate)
        new_w_glob[k] = torch.add(w_glob[k], server_step[k])
    return new_w_glob


# Error Rate based Rejection
# Fang, Minghong, Xiaoyu Cao, Jinyuan Jia, and Neil Gong.
# "Local model poisoning attacks to byzantine-robust federated learning."
# In 29th {USENIX} Security Symposium ({USENIX} Security 20), pp. 1605-1622. 2020.
def fed_err(w_locals, w_accs, w_glob, compromise_num, device):
    if len(w_locals) <= compromise_num:
        return w_glob
    for _ in range(compromise_num):
        index_min = min(range(len(w_accs)), key=w_accs.__getitem__)
        del w_accs[index_min]
        del w_locals[index_min]
    w_avg = {}
    for k in w_glob.keys():
        for w_local in w_locals:
            if k not in w_avg:
                w_avg[k] = torch.zeros_like(w_glob[k], device=device)
            w_avg[k] = torch.add(w_avg[k], w_local[k])
        w_avg[k] = torch.div(w_avg[k], len(w_locals))
    return w_avg


# Loss Function based Rejection
# Fang, Minghong, Xiaoyu Cao, Jinyuan Jia, and Neil Gong.
# "Local model poisoning attacks to byzantine-robust federated learning."
# In 29th {USENIX} Security Symposium ({USENIX} Security 20), pp. 1605-1622. 2020.
def fed_lfr(w_locals, w_losses, w_glob, compromise_num, device):
    if len(w_locals) <= compromise_num:
        return w_glob
    for _ in range(compromise_num):
        index_min = max(range(len(w_losses)), key=w_losses.__getitem__)
        del w_losses[index_min]
        del w_locals[index_min]
    w_avg = {}
    for k in w_glob.keys():
        for w_local in w_locals:
            if k not in w_avg:
                w_avg[k] = torch.zeros_like(w_glob[k], device=device)
            w_avg[k] = torch.add(w_avg[k], w_local[k])
        w_avg[k] = torch.div(w_avg[k], len(w_locals))
    return w_avg


# FLTrust
# Cao, Xiaoyu, Minghong Fang, Jia Liu, and Neil Zhenqiang Gong.
# "Fltrust: Byzantine-robust federated learning via trust bootstrapping."
# arXiv preprint arXiv:2012.13995 (2020).
def fed_trust(w_locals, w_glob, device):
    if len(w_locals) == 0:
        return w_glob
    similarity_dict = {}
    for idx, w_local in enumerate(w_locals):
        for k in w_glob.keys():
            if k not in similarity_dict:
                similarity_dict[k] = {}
            cos_sim = F.cosine_similarity(w_local[k], w_glob[k], dim=0)
            relu_cos_sim = F.relu(cos_sim)
            similarity_dict[k][idx] = relu_cos_sim

    w_avg = {}
    for k in w_glob.keys():
        for idx, w_local in enumerate(w_locals):
            if k not in w_avg:
                w_avg[k] = torch.zeros_like(w_glob[k], device=device)
            w_avg[k] = torch.add(w_avg[k], torch.mul(w_local[k], similarity_dict[k][idx]))
        w_avg[k] = torch.div(w_avg[k], sum(similarity_dict[k].values()))
    return w_avg

