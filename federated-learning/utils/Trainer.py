import logging

from utils.util import model_loader, test_model, train_model, record_log, disturb_w, extract_corrected_diff_sign, \
    extract_diff_sign, extract_ef_sign
from utils.Datasets import MyDataset

logger = logging.getLogger(__file__)


class Trainer:
    def __init__(self, uuid):
        self.net_local = None
        self.dataset = None
        self.uuid = uuid
        self.w_local = None
        self.w_local_acc = None
        self.w_local_loss = None
        # for sign SGD
        self.d_w_global = {}  # delta w_global
        self.net_momentum = {}  # momentum
        self.corrected_net_momentum = {}  # corrected momentum

    def init_dataset(self, dataset, dataset_train_size, iid, num_users):
        self.dataset = MyDataset(dataset, dataset_train_size, iid, num_users)
        if self.dataset.dataset_train is None:
            logger.error('Error: unrecognized dataset')
            return False
        return True

    def init_model(self, model, dataset, device):
        img_size = self.dataset.dataset_train[0][0].shape
        self.net_local = model_loader(model, dataset, device, img_size)
        if self.net_local is None:
            logger.error('Error: unrecognized model')
            return False
        return True

    def train_model(self, epoch, local_ep, device, lr, momentum, local_bs, poisoning_nodes):
        logger.debug("Train local model for user: {}, epoch: {}.".format(self.uuid, epoch))
        w_local, w_loss = train_model(self.net_local, self.dataset, self.uuid, local_ep, device, lr, momentum,
                                      local_bs, epoch == 0)
        w_local = self.poisoning_attack(w_local, poisoning_nodes)
        self.w_local = w_local

    def extract_corrected_sign(self, w_global, sign_sgd_beta, d_w_global, device):
        self.w_local = extract_corrected_diff_sign(
            self.w_local, w_global, self.net_momentum, self.corrected_net_momentum, sign_sgd_beta, d_w_global, device
        )

    def extract_sign(self, w_global, sign_sgd_beta, device):
        self.w_local = extract_diff_sign(self.w_local, w_global, self.net_momentum, sign_sgd_beta, device)

    def extract_ef_sign(self, w_global, d_w_global, device):
        self.w_local = extract_ef_sign(self.w_local, w_global, self.corrected_net_momentum, d_w_global, device)

    def evaluate_model(self, iid, local_test_bs, device):
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = \
            test_model(self.net_local, self.dataset, self.uuid, iid, local_test_bs, device)
        return acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4

    def evaluate_model_loss(self, iid, local_test_bs, device):
        loss_local, loss_local_skew1, loss_local_skew2, loss_local_skew3, loss_local_skew4 = \
            test_model(self.net_local, self.dataset, self.uuid, iid, local_test_bs, device, get_acc=False)
        return loss_local, loss_local_skew1, loss_local_skew2, loss_local_skew3, loss_local_skew4

    def evaluate_model_with_log(self, iid, local_test_bs, device, epoch, clean=False):
        acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4 = self.evaluate_model(
            iid, local_test_bs, device)
        record_log(self.uuid, epoch,
                   [acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4], clean=clean)
        return acc_local, acc_local_skew1, acc_local_skew2, acc_local_skew3, acc_local_skew4

    def poisoning_attack(self, w_local, poisoning_nodes):
        # fake attackers
        poisoning_nodes_str = poisoning_nodes
        if len(poisoning_nodes_str) < 1:
            # if the parameter is empty, do nothing
            return w_local
        poisoning_nodes = list(map(int, list(poisoning_nodes_str.split(","))))
        if int(self.uuid) in poisoning_nodes:
            logger.debug("As a poisoning attacker ({}), manipulate local gradients!".format(poisoning_nodes))
            w_local = disturb_w(w_local)
        return w_local

