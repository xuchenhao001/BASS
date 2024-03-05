import logging

from utils.EnvStore import EnvStore
from utils.Trainer import Trainer


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

    for epoch in range(env_store.args.epochs):
        logger.debug(f"# start epoch {epoch} #")
        # local training, the trained local models are stored in trainer.w_local
        for trainer in trainer_store:
            trainer.train_model(
                epoch, env_store.args.local_ep, env_store.args.device, env_store.args.lr,
                env_store.args.momentum, env_store.args.local_bs, env_store.args.poisoning_nodes
            )
            trainer.net_local.load_state_dict(trainer.w_local)
            trainer.evaluate_model_with_log(
                env_store.args.iid, env_store.args.local_test_bs, env_store.args.device, epoch
            )
    logger.info("########## ALL DONE! ##########")


if __name__ == "__main__":
    main()
