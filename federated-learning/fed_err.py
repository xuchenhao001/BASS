import logging
import random

from utils.EnvStore import EnvStore
from utils.Trainer import Trainer
from utils.util import calculate_server_step
from models.Fed import fed_err


logger = logging.getLogger(__file__)


env_store = EnvStore()
trainer_store = []


def init_trainer(uuid):
    trainer = Trainer(uuid)
    load_result = trainer.init_dataset(
        env_store.args.dataset, env_store.args.dataset_train_size, env_store.args.iid, env_store.args.num_users
    )
    if not load_result:
        exit(0)
    trainer.dataset.init_trojan_attack(env_store.args)
    load_result = trainer.init_model(env_store.args.model, env_store.args.dataset, env_store.args.device)
    if not load_result:
        exit(0)
    return trainer


def main():
    # init environment arguments
    env_store.init()
    logging.basicConfig(level=env_store.args.log_level)

    # init trainers
    for uuid in range(env_store.args.num_users):
        logger.debug("start new client")
        new_trainer = init_trainer(uuid)
        trainer_store.append(new_trainer)

    # init global models for all clients
    w_global = trainer_store[0].net_local.state_dict()
    for trainer in trainer_store:
        trainer.net_local.load_state_dict(w_global)
    # for sign SGD
    d_w_global = {}  # delta w_global

    for epoch in range(env_store.args.epochs):
        logger.debug(f"# start epoch {epoch} #")
        # local training, the trained local models are stored in trainer.w_local
        for trainer in trainer_store:
            trainer.train_model(
                epoch, env_store.args.local_ep, env_store.args.device, env_store.args.lr,
                env_store.args.momentum, env_store.args.local_bs, env_store.args.poisoning_nodes
            )
            if env_store.args.sign_sgd:
                trainer.extract_corrected_sign(
                    w_global, env_store.args.sign_sgd_beta, d_w_global, env_store.args.device
                )

        # received trainers after DDoS
        if env_store.args.ddos_attack:
            # mimic DDoS attacks here
            response_num = int((1 - env_store.args.ddos_no_response_percent) * env_store.args.num_users)
            received_trainers = random.choices(trainer_store, k=response_num)
        else:
            received_trainers = trainer_store

        # aggregate global model
        logger.debug("Gathered enough local models, average their weights")
        w_locals = []
        acc_w_locals = []
        for trainer in received_trainers:
            # evaluate local model accuracy first
            acc, _, _, _, _ = trainer.evaluate_model(
                env_store.args.iid, env_store.args.local_test_bs, env_store.args.device
            )
            acc_w_locals.append(acc)
            w_locals.append(trainer.w_local)
        client_compromise_num = round(env_store.args.err_compromise_rate * env_store.args.num_users)
        new_w_global = fed_err(w_locals, acc_w_locals, w_global, client_compromise_num, env_store.args.device)

        # update server step
        d_w_global = calculate_server_step(w_global, new_w_global)

        # finally, evaluate the global model
        w_global = new_w_global
        for trainer in trainer_store:
            trainer.net_local.load_state_dict(w_global)
            trainer.evaluate_model_with_log(
                env_store.args.iid, env_store.args.local_test_bs, env_store.args.device, epoch
            )
    logger.info("########## ALL DONE! ##########")


if __name__ == "__main__":
    main()
