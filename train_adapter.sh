#!/bin/bash
# Options are follwed in the paper and /mnt/hlilabshare/HLILab_Public/yjspecial/yjspecial/everest/EmotionalBART/output/comparison between r of lora/[0] ours/BART16:lora/[Adapter_lora_r=8] [Resume checkpoints_ours_BART16] [Emotional_softmax/0.99/2.0] BART-LARGE BZ16 FP16 weight_0.5/rerun.sh


GPU=$1
OUTPUT=$2
plm=$3

weight=0.5
loss_type=softmax
loss_beta=0.99
loss_gamma=2.0
adapter_type=lora

is_fp16=true
rs=(q#v)

mkdir -p $OUTPUT
for r in "${rs[@]}"; do
CUDA_VISIBLE_DEVICES=$GPU python src/main.py \
    --fp16 $is_fp16 \
    --gpu_index $GPU \
    --seed 7777 \
    --do_train \
    --num_train_epochs 15 \
    --data_dir data \
    --resume_from_checkpoint $plm \
    --output_dir $OUTPUT \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --save_strategy steps \
    --report_to wandb \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --predict_with_generate \
    --metric_for_best_model rougeL \
    --load_best_model_at_end \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --max_source_length 512 \
    --max_target_length 64 \
    --generation_max_length 64 \
    --generation_num_beams 5 \
    --remove_unused_columns false \
    --check_emotional_consistency \
    --train_adapter \
    --adapter_config $adapter_type \
    --adapter_non_linearity $r \
    --adapter_reduction_factor 2 \
    --enable_logging false \
    --emotional_loss_weight $weight \
    --emotional_loss_type $loss_type \
    --emotional_loss_beta $loss_beta \
    --emotional_loss_gamma $loss_gamma
done
