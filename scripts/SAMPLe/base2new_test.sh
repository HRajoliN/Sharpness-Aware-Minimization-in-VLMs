#!/bin/bash

# custom config
DATA="DATA"
# $NORMALIZATION=False
TRAINER=SAMPLe

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
LOADEP=20
SUB=new

COMMON_DIR=${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}/${ENBLR}/rho=${RHO}/alpha=${ALPHA}/lambda=${LAMBDA}/LR=${LR}

if [ "$NORMALIZATION" = False ]; then
    # MODEL_DIR=output/base2new/not-normal/train_base/${COMMON_DIR}
    MODEL_DIR="/home/fafghah/Documents/Hossein Rajoli/SAMPLe/output/base2new/not-normal/train_base/imagenet/SAMPLe/vit_b16_c2_ep20_batch512_4+4ctx_16shots/nctx4_cscFalse_ctpend/seed1/SAMPLe/rho=0.1/alpha=0.001/lambda=0.015/LR=0.05"
    DIR=output/base2new/not-normal/test_${SUB}/${COMMON_DIR}
else
    # MODEL_DIR=output/base2new/normal/train_base/${COMMON_DIR}
    MODEL_DIR="/home/fafghah/Documents/Hossein Rajoli/SAMPLe/output/base2new/not-normal/train_base/imagenet/SAMPLe/vit_b16_c2_ep20_batch512_4+4ctx_16shots/nctx4_cscFalse_ctpend/seed1/SAMPLe/rho=0.1/alpha=0.001/lambda=0.015/LR=0.05"
    DIR=output/base2new/normal/test_${SUB}/${COMMON_DIR}
fi    

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
