import logging
import random
import time
import torch

from models.Nets import CNNCifar, CNNMnist, CNNFashion, UCI_CNN, MLP, ResNet18
from models.local_test import test_img_total
from models.local_train import train_cnn_mlp

logger = logging.getLogger(__file__)


def model_loader(model_name, dataset_name, device, img_size):
    net_glob = None
    # build model, init part
    if model_name == 'cnn' and dataset_name == 'cifar':
        net_glob = CNNCifar(num_classes=10).to(device)
    elif model_name == 'cnn' and dataset_name == 'mnist':
        net_glob = CNNMnist(num_classes=10).to(device)
    elif model_name == 'cnn' and dataset_name == 'fmnist':
        net_glob = CNNFashion(num_classes=10).to(device)
    elif model_name == 'cnn' and dataset_name == 'uci':
        net_glob = UCI_CNN(num_classes=6).to(device)
    elif model_name == 'cnn' and dataset_name == 'realworld':
        net_glob = UCI_CNN(num_classes=8).to(device)
    elif model_name == 'resnet18' and dataset_name == 'cifar':
        net_glob = ResNet18().to(device)
    elif model_name == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in).to(device)
    return net_glob


def test_model(net_glob, my_dataset, idx, is_iid, local_test_bs, device, get_acc=True):
    if is_iid:
        idx_total = [my_dataset.test_users[idx]]
        acc_list, loss_list = test_img_total(net_glob, my_dataset, idx_total, local_test_bs, device)
        acc_local = acc_list[0]
        loss_local = loss_list[0]
        if get_acc:
            return acc_local, 0.0, 0.0, 0.0, 0.0
        else:
            return loss_local, 0.0, 0.0, 0.0, 0.0
    else:
        idx_total = [my_dataset.test_users[idx], my_dataset.skew_users[0][idx], my_dataset.skew_users[1][idx],
                     my_dataset.skew_users[2][idx], my_dataset.skew_users[3][idx]]
        acc_list, loss_list = test_img_total(net_glob, my_dataset, idx_total, local_test_bs, device)
        acc_local = acc_list[0]
        acc_local_skew1 = acc_list[1]
        acc_local_skew2 = acc_list[2]
        acc_local_skew3 = acc_list[3]
        acc_local_skew4 = acc_list[4]
        loss_local = loss_list[0]
        loss_local_skew1 = loss_list[1]
        loss_local_skew2 = loss_list[2]
        loss_local_skew3 = loss_list[3]
        loss_local_skew4 = loss_list[4]
        if get_acc:
            return acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4
        else:
            return loss_local, loss_local_skew1, loss_local_skew2, loss_local_skew3, loss_local_skew4


def train_model(net_glob, my_dataset, idx, local_ep, device, lr, momentum, local_bs, is_first_epoch):
    return train_cnn_mlp(net_glob, my_dataset, idx, local_ep, device, lr, momentum, local_bs, is_first_epoch)


def extract_diff_sign(w_local, w_glob, momentum, beta, device):
    d_w_local = {}
    sign = {}
    for k in w_local.keys():
        d_w_local[k] = torch.sub(w_local[k], w_glob[k])
        # initialize momentum with zero
        if k not in momentum:
            momentum[k] = torch.zeros_like(d_w_local[k], device=device)
        momentum[k] = torch.add(torch.mul(momentum[k], beta), torch.mul(d_w_local[k], 1-beta))
        sign[k] = torch.sign(momentum[k])
    return sign


def extract_corrected_diff_sign(w_local, w_global, momentum, corrected_momentum, beta, d_w_global, device):
    d_w_local = {}
    sign = {}
    residual_error = {}
    for k in w_local.keys():
        d_w_local[k] = torch.sub(w_local[k], w_global[k])
        # initialize momentum with zero
        if k not in momentum:
            momentum[k] = torch.zeros_like(d_w_local[k], device=device)
        # initialize corrected momentum with zero
        if k not in corrected_momentum:
            corrected_momentum[k] = torch.zeros_like(d_w_local[k], device=device)
        # initialize delta w_local with zero
        if k not in d_w_global:
            d_w_global[k] = torch.zeros_like(d_w_local[k], device=device)
        # residual_error[k] = torch.sub(corrected_momentum[k], d_w_global[k])  # cumulated residual error
        residual_error[k] = torch.sub(momentum[k], d_w_global[k])  # non-cumulated residual error
        momentum[k] = torch.add(torch.mul(momentum[k], beta), torch.mul(d_w_local[k], 1-beta))
        corrected_momentum[k] = torch.add(momentum[k], residual_error[k])
        sign[k] = torch.sign(corrected_momentum[k])
    return sign


def extract_ef_sign(w_local, w_glob, corrected_momentum, d_w_global, device):
    d_w_local = {}
    scaled_sign = {}
    residual_error = {}
    for k in w_local.keys():
        # gt := stochasticGradient(xt)
        d_w_local[k] = torch.sub(w_local[k], w_glob[k])
        # initialize corrected momentum with zero
        if k not in corrected_momentum:
            corrected_momentum[k] = torch.zeros_like(d_w_local[k], device=device)
        # initialize delta w_local with zero
        if k not in d_w_global:
            d_w_global[k] = torch.zeros_like(d_w_local[k], device=device)
        residual_error[k] = torch.sub(corrected_momentum[k], d_w_global[k])
        # pt := \gamma gt + et (corrected_momentum is pt)
        corrected_momentum[k] = torch.add(d_w_local[k], residual_error[k]).to(torch.float)
        # cumulate scaling ||pt||_1 and d
        pt_norm_1 = torch.linalg.norm(torch.flatten(corrected_momentum[k]), 1, dim=0)
        scaling = torch.div(pt_norm_1, torch.numel(corrected_momentum[k]))
        scaled_sign[k] = torch.sign(corrected_momentum[k]) * scaling
    return scaled_sign


def disturb_w(w):
    disturbed_w = {}
    for key, param in w.items():
        beta = (random.random()-0.5)*2  # disturb value in range [-1, 1)
        transformed_w = param * beta
        disturbed_w[key] = transformed_w
    return disturbed_w


# for dynamic adjusting server learning rate by multiply 0.1 in every 20 rounds of training
def server_learning_rate_adjust(current_epoch, server_lr_decimate, server_lr):
    server_lr_decimate_str = server_lr_decimate.strip()
    if len(server_lr_decimate_str) < 1:
        # if the parameter is empty, do nothing
        return server_lr
    server_lr_decimate = list(map(int, list(server_lr_decimate_str.split(","))))
    if int(current_epoch) in server_lr_decimate:
        server_lr *= 0.1
        logger.info("Decimate the server learning rate to: {}.".format(server_lr))
    return server_lr


def calculate_server_step(old_w_glob, new_w_glob):
    d_w_global = {}
    for k in old_w_glob.keys():
        d_w_global[k] = torch.sub(new_w_glob[k], old_w_glob[k])
    return d_w_global


# time_list: [total_time, round_time, train_time, test_time]
# acc_list: [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4]  (for cnn or mlp)
def record_log(user_id, epoch, acc_list, clean=False):
    filename = "result-record_" + str(user_id) + ".txt"

    # first time clean the file
    if clean:
        open(filename, 'w').close()

    with open(filename, "a") as time_record_file:
        current_time = time.strftime("%H:%M:%S", time.localtime())
        time_record_file.write(current_time + "[" + "{:03d}".format(epoch) + "]"
                               + " <acc_local> " + str(acc_list[0])[:8]
                               + " <acc_local_skew1> " + str(acc_list[1])[:8]
                               + " <acc_local_skew2> " + str(acc_list[2])[:8]
                               + " <acc_local_skew3> " + str(acc_list[3])[:8]
                               + " <acc_local_skew4> " + str(acc_list[4])[:8]
                               + "\n")

