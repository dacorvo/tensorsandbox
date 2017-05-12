#!/bin/bash

model=${1:-'cs231n'}
steps=${2:-10000}

SCRIPTPATH=$(readlink -f ${0})
SCRIPTDIR=$(dirname ${SCRIPTPATH})

args="--model=${model}"

# if USE_AUGMENTATION is set we use data augmentation for training
if [ ${USE_AUGMENTATION+x} ]; then
    args="${args} --data_aug"
fi

train_args="${args} --max_steps=${steps}"
eval_args="${args}"

# Launch training
python ${SCRIPTDIR}/train.py ${train_args} &

# Wait a little to let it start
sleep 5
# Launch eval on train data
python ${SCRIPTDIR}/eval.py ${eval_args} --eval_data train &
# Launch eval on test data
python ${SCRIPTDIR}/eval.py ${eval_args}
