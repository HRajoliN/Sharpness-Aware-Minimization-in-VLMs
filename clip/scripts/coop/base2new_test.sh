#!/bin/bash

# custom config
DATA="DATA"

TRAINER=CoOp

DATASET=$1
CFG=$2  # config file  (vit_b16_c2_ep20_batch4_4+4ctx)
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
SEED=${7}

LOADEP=20
SUB=new

# COMMON_DIR="oxford_flowers/SAMPLe/vit_b16_c2_ep20_batch4_4+4ctx_16shots/nctx4_cscFalse_ctpend/seed1/SAM/rho=0.05/alpha=0.5/lambda=0.5"
# COMMON_DIR="oxford_flowers/SAMPLe/vit_b16_c2_ep20_batch4_4+4ctx_16shots/nctx4_cscFalse_ctpend/seed1/None/rho=0.00/alpha=0.00/lambda=0.00"
COMMON_DIR=${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming evaluation..."
    
    # If you want to exit here and not re-evaluate when results exist, uncomment the next line:
    # exit 0

    python3 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Running evaluation and saving the output to ${DIR}..."

    python3 train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
