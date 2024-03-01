#! bash/bin

file=$1

device=$2
checkpoint=$3

CUDA_VISIBLE_DEVICES=$device python $file --model_path $checkpoint