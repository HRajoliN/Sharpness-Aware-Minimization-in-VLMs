#!/bin/bash

# custom config
DATA=./DATA
TRAINER=CoOp
NORMALIZATION=False
ADAPTIVE=False



DATASET=$1
CFG=$2  # config file  (vit_b16_c2_ep20_batch4_4+4ctx)
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)

SEED=${7}

# CFG=vit_b16_c2_ep20_batch4_4+4ctx
SHOTS=16
SUB=base

DIR=output/base2new/train_${SUB}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

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
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
