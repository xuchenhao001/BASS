# BASS

BASS: A Blockchain-Based Asynchronous SignSGD Architecture for Efficient and Secure Federated Learning

## Install

How to install this project on your operating system.

### Prerequisite

* Ubuntu 22.04

* Python 3.10.6

* The BASS project should be cloned into the home directory, like `~/BASS`.

Install dependencies:

```bash
pip3 install -r requirements.txt
```

## Run

How to start & stop this project.

The parameters for the training are at `./BASS/federated-learning/utils/options.py`

```bash
cd federated-learning/
rm -f result-*
python3 fed_bass.py
# Or start in background
nohup python3 -u fed_bass.py > fed_bass.log 2>&1 &
```

# Comparative Experiments

The comparative experiments include (under `BEFS/federated-learning/` directory):

```bash
fed_bass.py  # BASS
fed_avg.py  # FedAvg
fed_ecsign.py  # EC-signSGD
fed_efsign.py  # EF-signSGD
fed_err.py  # ERR-FedAvg
fed_lfr.py  # LFR-FedAvg
fed_mvsign.py  # MV-signSGD
fed_rlrsign.py  # RLR-signSGD
fed_fleam.py  # FLEAM
fed_trust.py  # FLTrust
local_train.py  # Local
```
