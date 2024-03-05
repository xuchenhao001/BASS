import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=512, help="local batch size: B")
    parser.add_argument('--local_test_bs', type=int, default=512, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # model arguments, support model: "cnn", "mlp", "resnet18"
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    # support dataset: "mnist", "fmnist", "cifar", "uci", "realworld"
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--log_level', type=str, default='DEBUG', help='DEBUG, INFO, WARNING, ERROR, or CRITICAL')

    # for APFL
    parser.add_argument('--apfl_hyper', type=float, default=0.3, help='APFL hypermeter alpha')

    # customized parameters
    # total dataset training size: MNIST: 60000, FASHION-MNIST:60000, CIFAR-10: 60000, UCI: 10929, REALWORLD: 285148,
    parser.add_argument('--dataset_train_size', type=int, default=1500, help="total dataset training size")
    # ip address that is used to test local IP
    parser.add_argument('--test_ip_addr', type=str, default="10.150.187.13", help="ip address used to test local IP")
    # sign SGD, default is false
    parser.add_argument('--sign_sgd', action='store_true', help='whether adopting sign SGD or not')
    parser.add_argument('--server_lr', type=float, default=0.01, help="sign SGD server learning rate")
    parser.add_argument('--server_lr_decimate', type=str, default='100,150',
                        help='comma-separated epoch number (int) that learning rate to be divided by 10')
    parser.add_argument('--sign_sgd_beta', type=float, default=0.5, help="beta parameter in sign SGD momentum")
    # Error Rate based Rejection and Loss Function based Rejection
    parser.add_argument('--err_compromise_rate', type=float, default=0.3, help="compromise rate for ERR and LFR")
    parser.add_argument('--fed_listen_port', type=int, default=8888, help="federated learning server listening port")

    # security-related parameters
    # poisoning attacker ids, must be string type. For example, "0,1,2" or ""
    parser.add_argument('--poisoning_nodes', type=str, default="", help="id of nodes that will launch trojan attack")

    # launch ddos attack or not
    parser.add_argument("--ddos_attack", action='store_true', help="launch ddos attack or not")
    # under ddos attack, no response request percent
    parser.add_argument("--ddos_no_response_percent", type=float, default=0.9)

    # backdoor attack (trojan attack)
    parser.add_argument('--trojan_nodes', type=str, default='', help="id of nodes that will launch trojan attack")
    parser.add_argument('--trojan_base_class', type=int, default=1, help="base class for trojan attack")
    parser.add_argument('--trojan_target_class', type=int, default=3, help="target class for trojan attack")
    parser.add_argument('--trojan_frac', type=float, default=0.0, help="fraction of trojan dataset")

    args = parser.parse_args()
    return args
