#!/bin/bash

# custom config
DATA=./DATA
TRAINER=SAMPLe
# NORMALIZATION=True
ADAPTIVE=False



DATASET=$1
CFG=$2  # config file  (vit_b16_c2_ep20_batch4_4+4ctx)
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)

ENBLR=$7
RHO=$8      # defines the radii of SAM perturbation
ALPHA=$9    # defines the weight of batch-specific noisy perturbation
LAMBDA=${10}
SEED=${11}
NORMALIZATION=${12}
LR=${13}


# DATASET=food101
# CFG=vit_b16_c2_ep20_batch4_4+4ctx  # config file  (vit_b16_c2_ep20_batch4_4+4ctx)
# CTP=end  # class token position (end or middle)
# NCTX=4  # number of context tokens
# SHOTS=16  # number of shots (1, 2, 4, 8, 16)
# CSC=False  # class-specific context (False or True)

# ENBLR=SAMPLe
# RHO=0.1      # defines the radii of SAM perturbation
# ALPHA=0.005    # defines the weight of batch-specific noisy perturbation
# LAMBDA=0.2
# SEED=1
# NORMALIZATION=False


# CFG=vit_b16_c2_ep20_batch4_4+4ctx
# SHOTS=16
SUB=base

if [ "$NORMALIZATION" = False ]; then
    DIR=output/base2new/not-normal/train_${SUB}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}/${ENBLR}/rho=${RHO}/alpha=${ALPHA}/lambda=${LAMBDA}/LR=${LR}
else
    DIR=output/base2new/normal/train_${SUB}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}/${ENBLR}/rho=${RHO}/alpha=${ALPHA}/lambda=${LAMBDA}/LR=${LR}
fi

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Exiting..."
    # exit 0   # Exits the script if the directory exists
        # echo "Running this job and saving the output to ${DIR}..."
    python3 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.SAMPLe.NORMALIZATION ${NORMALIZATION} \
    TRAINER.SAMPLe.ADAPTIVE ${ADAPTIVE} \
    TRAINER.SAMPLe.ENBLR ${ENBLR} \
    TRAINER.SAMPLe.RHO ${RHO} \
    TRAINER.SAMPLe.ALPHA ${ALPHA} \
    TRAINER.SAMPLe.EMA_LAMBDA ${LAMBDA}\
    OPTIM.LR ${LR}\
    DATASET.SUBSAMPLE_CLASSES ${SUB}
else
    echo "Running this job and saving the output to ${DIR}..."
    python3 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.SAMPLe.NORMALIZATION ${NORMALIZATION} \
    TRAINER.SAMPLe.ADAPTIVE ${ADAPTIVE} \
    TRAINER.SAMPLe.ENBLR ${ENBLR} \
    TRAINER.SAMPLe.RHO ${RHO} \
    TRAINER.SAMPLe.ALPHA ${ALPHA} \
    TRAINER.SAMPLe.EMA_LAMBDA ${LAMBDA}\
    OPTIM.LR ${LR}\
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
