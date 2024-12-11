#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate

# Common LoRA modules:
# LLaMA/Baichuan: down_proj,W_pack,gate_proj,up_proj,o_proj
# Opt: self_attn,fc1,fc2
# Qwen: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

train_data="./data/MathInstruct.json"
eval_data="./data/gsm8k/train.json"

OUTPUT_PATH=./output
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH

deepspeed --include localhost:0,1 meta-lora-multigpu.py \
   --train_data $train_data \
   --dataset_format alpaca \
   --eval_data $eval_data \
   --eval_dataset_format gsm8k \
   --model_name_or_path /data/lwc/Baichuan2-main/Baichuan2-7B-Base \
   --meta True \
   --dtype bf16 \
   --local_rank -1 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 400 \
   --source_max_len 200 \
   --target_max_len 200 \
   --learning_rate 5e-4 \
   --weight_decay 0 \
   --num_train_epochs 1 \
   --max_steps 500 \
   --gradient_accumulation_steps 1 \
   --validation_interval 10 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --offload \
   --lora_dim 32 \
   --lora_module_name down_proj,W_pack,gate_proj,up_proj,o_proj \
   --only_optimize_lora \
   --gradient_checkpointing \
   --deepspeed \
   --print_loss \
   --output_dir $OUTPUT_PATH \
   > $OUTPUT_PATH/train.log
