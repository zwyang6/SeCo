#! bash/bin

file=$1

nproc_per_node=1
master_port=$2
device_gpu=$3
exp_des=$4

CUDA_VISIBLE_DEVICES=$device_gpu python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port $file --log_tag=$exp_des