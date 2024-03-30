#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=5,6,7
DIR=`pwd`

GPUS_PER_NODE=3
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6004

MODEL="/home/bailong/.cache/modelscope/hub/qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="data/jsons/qwen_train.json"

LORA_GROUP_SIZE=0 # Use GALora if > 1; default to 0. Must be also divisible by Linear Layer's in_features
LORA_MAMBA_DEPTH=1 # Use Mamba if > 1; default to 0. Only supported on GALora

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune_galora.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --fix_vit True \
    --output_dir output_qwen_galora \
    --num_train_epochs 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --use_lora \
    --gradient_checkpointing \
    --deepspeed finetune/ds_config_zero2.json \
    --lora_group_size $LORA_GROUP_SIZE \
    --lora_mamba_depth $LORA_MAMBA_DEPTH