import logging
import torch

from utils.options import args_parser


logger = logging.getLogger(__file__)


class EnvStore:
    def __init__(self):
        self.args = None

    def init(self):
        self.args = args_parser()
        self.args.device = torch.device(
            'cuda:{}'.format(self.args.gpu) if torch.cuda.is_available() and self.args.gpu != -1 else 'cpu')
        # print parameters in log
        arguments = vars(self.args)
        logger.info("==========================================")
        for k, v in arguments.items():
            arg = "{}: {}".format(k, v)
            logger.info("* {0:<40}".format(arg))
        logger.info("==========================================")
