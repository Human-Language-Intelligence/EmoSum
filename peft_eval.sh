#!/bin/bash

GPU=$1
BASE_CHECKPOINT=$2
TARGET_CHECKPOINT=$3
OUTPUT=$4

weight=0.5

today=$(date "+%y%m%d")
tag=summary1

is_fp16=true
fp_value=16


CUDA_VISIBLE_DEVICES=$GPU python src/main.py \
    --fp16 $is_fp16 \
    --gpu_index $GPU \
    --seed 7777 \
    --num_train_epochs 15 \
    --data_dir data \
    --resume_from_checkpoint $BASE_CHECKPOINT \
    --output_dir $OUTPUT \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --save_strategy epoch \
    --warmup_steps 100 \
    --report_to wandb \
    --evaluation_strategy epoch \
    --predict_with_generate \
    --metric_for_best_model rougeL \
    --load_best_model_at_end \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --test_label_tag $tag \
    --max_source_length 512 \
    --max_target_length 64 \
    --generation_max_length 64 \
    --generation_num_beams 5 \
    --remove_unused_columns false \
    --check_emotional_consistency \
    --enable_logging false \
    --emotional_loss_weight $weight \
    --train_adapter \
    --adapter_config lora \
    --load_adapter "$TARGET_CHECKPOINT" \
    --adapter_non_linearity swish
