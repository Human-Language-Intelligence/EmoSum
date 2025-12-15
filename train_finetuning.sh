#!/bin/bash
# Options are follwed in the paper and /mnt/hlilabshare/HLILab_Public/yjspecial/yjspecial/everest/EmotionalBART/output/[EVAL_Base] 230418/EmotionalFineTuning/BART16:summary1/[Resume checkpoints_emotional_BART16] [Emotional_cross_entropy/0.9999/2.0] BART-LARGE BZ32 FP16 weight_0.5/rerun.sh

GPU=$1
OUTPUT=$2

model="./bart-large"
# model=BART16
weight=0.5
loss_type=cb_softmax
loss_beta=0.99
loss_gamma=2.0

today=$(date "+%y%m%d")
batch_size=32
is_fp16=true
model_name="${model}"


mkdir -p $OUTPUT

CUDA_VISIBLE_DEVICES=$GPU python src/main.py \
    --fp16 $is_fp16 \
    --gpu_index $GPU \
    --seed 42 \
    --do_train \
    --num_train_epochs 15 \
    --data_dir data \
    --output_dir $OUTPUT \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --save_strategy epoch \
    --report_to wandb \
    --evaluation_strategy epoch \
    --eval_steps 100 \
    --predict_with_generate \
    --metric_for_best_model rougeL \
    --load_best_model_at_end \
    --model_name $model_name \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size 4 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --max_source_length 512 \
    --max_target_length 100 \
    --generation_max_length 100 \
    --generation_num_beams 5 \
    --remove_unused_columns false \
    --check_emotional_consistency \
    --enable_logging false \
    --emotional_loss_weight $weight \
    --emotional_loss_type $loss_type \
    --emotional_loss_beta $loss_beta \
    --emotional_loss_gamma $loss_gamma
