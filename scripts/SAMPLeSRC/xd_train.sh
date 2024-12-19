#!/bin/bash

#cd ../..

# custom config
DATA=./DATA
TRAINER=SAMPLeSRC
NORMALIZATION=False
ADAPTIVE=False

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
# CSC=False
ENBLR=$7
RHO=$8      # defines the radii of SAM perturbation
ALPHA=$9    # defines the weight of batch-specific noisy perturbation
LAMBDA=${10}
SEED=${11}
NORMALIZATION=${12}
LR=${13}



# for SEED in 1 2 3
# do
#     DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
#     if [ -d "$DIR" ]; then
#         echo "Results are available in ${DIR}. Skip this job"
#     else
#         echo "Run this job and save the output to ${DIR}"
#         python3 train.py \
#         --root ${DATA} \
#         --seed ${SEED} \
#         --trainer ${TRAINER} \
#         --dataset-config-file configs/datasets/${DATASET}.yaml \
#         --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#         --output-dir ${DIR} \
#         TRAINER.COOP.N_CTX ${NCTX} \
#         TRAINER.COOP.CSC ${CSC} \
#         TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
#         DATASET.NUM_SHOTS ${SHOTS}
#     fi
# done


# for SEED in 
# do
SUB=all

DIR=output/Cross_Dataset/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}/${ENBLR}/rho=${RHO}/alpha=${ALPHA}/lambda=${LAMBDA}/lr=${LR}
echo "Run this job and save the output to ${DIR}"
python3 train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
TRAINER.SAMPLe.N_CTX ${NCTX} \
TRAINER.SAMPLe.CSC ${CSC} \
TRAINER.SAMPLe.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} \
TRAINER.SAMPLe.NORMALIZATION ${NORMALIZATION} \
TRAINER.SAMPLe.ADAPTIVE ${ADAPTIVE} \
TRAINER.SAMPLe.ENBLR ${ENBLR} \
TRAINER.SAMPLe.RHO ${RHO} \
TRAINER.SAMPLe.ALPHA ${ALPHA} \
TRAINER.SAMPLe.EMA_LAMBDA ${LAMBDA} \
OPTIM.LR ${LR} \
DATASET.SUBSAMPLE_CLASSES ${SUB}
# done
