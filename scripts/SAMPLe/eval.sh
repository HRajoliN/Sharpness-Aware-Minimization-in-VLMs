#!/bin/bash

#cd ../..

# custom config
DATA="./DATA"
TRAINER=SAMPLe
SHOTS=16
NCTX=16
CSC=False
CTP=end
NORMALIZATION=False
ADAPTIVE=False

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
ENBLR=$7
RHO=$8      # defines the radii of SAM perturbation
ALPHA=$9    # defines the weight of batch-specific noisy perturbation
LAMBDA=${10}
# SEED=${11}

dirr="output/oxford_flowers/SAMPLe/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/None/rho=0.05/alpha=0.5/lambda=0.5"
SUB=all

for SEED in 1
do
    python3 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${DATASET}/seed${SEED} \
    --model-dir ${dirr} \
    --load-epoch 50 \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.SAMPLe.NORMALIZATION ${NORMALIZATION} \
    TRAINER.SAMPLe.ADAPTIVE ${ADAPTIVE} \
    TRAINER.SAMPLe.ENBLR ${ENBLR} \
    TRAINER.SAMPLe.RHO ${RHO} \
    TRAINER.SAMPLe.ALPHA ${ALPHA} \
    TRAINER.SAMPLe.EMA_LAMBDA ${LAMBDA} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
done

