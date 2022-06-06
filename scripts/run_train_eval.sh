#!/bin/bash

# parameters
epsilon=8e5
delta=1e-1
alpha=1e-1
iter_num=1e6
timestamp=`date '+%s'`

# preparation
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd ${SHELL_FOLDER}/../
if [ ! -d "./results" ]; then
    mkdir ./results
fi

python train_eval.py --epsilon=${epsilon} --delta=${delta} --alpha=${alpha} --iter_num=${iter_num}| tee ./results/"${timestamp}_epsilon_${epsilon}_delta_${delta}_alpha_${alpha}_iter_num_${iter_num}".txt
