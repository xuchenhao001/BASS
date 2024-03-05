#!/bin/bash

PROJ_PATH=$PWD

function arrangeOutput(){
    local DIR_NAME=$1

    mkdir -p $PROJ_PATH/output/

    rm -rf $PROJ_PATH/temp/
    mkdir -p $PROJ_PATH/temp/
    cp $PROJ_PATH/federated-learning/result-*.txt $PROJ_PATH/temp/
    cp $PROJ_PATH/server.log $PROJ_PATH/temp/
    mv $PROJ_PATH/temp $PROJ_PATH/output/${DIR_NAME}

    rm -f $PROJ_PATH/federated-learning/result-*.txt
}

function main() {
  cd federated-learning/
  python3 fed_bass.py --model="resnet18" --dataset="cifar" --dataset_train_size=50000 --iid --sign_sgd --num_users=1
  arrangeOutput "resnet18_cifar_fed_bass"
  python3 fed_avg.py --model="resnet18" --dataset="cifar" --dataset_train_size=50000 --iid --num_users=1
  arrangeOutput "resnet18_cifar_fed_avg"
  python3 fed_efsign.py --model="resnet18" --dataset="cifar" --dataset_train_size=50000 --iid --num_users=1
  arrangeOutput "resnet18_cifar_fed_efsign"
  python3 fed_mvsign.py --model="resnet18" --dataset="cifar" --dataset_train_size=50000 --iid --num_users=
  arrangeOutput "resnet18_cifar_fed_mvsign"
  python3 fed_avg.py --model="resnet18" --dataset="cifar" --dataset_train_size=50000 --iid --num_users=1
  arrangeOutput "resnet18_cifar_fed_dtwn"
  python3 local_train.py --model="resnet18" --dataset="cifar" --dataset_train_size=50000 --iid --num_users=1
  arrangeOutput "resnet18_cifar_local_train"
}

main > script.log
