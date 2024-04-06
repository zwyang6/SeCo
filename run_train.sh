#! /bin/bash

file=$1
nproc_per_node=$2
master_port=$3
device_gpu=$4
exp_des=$5

CUDA_VISIBLE_DEVICES=$device_gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file --log_tag=$exp_des
