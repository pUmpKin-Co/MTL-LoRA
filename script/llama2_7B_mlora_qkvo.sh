SCRIPT_PATH=mlora_finetune.py
DATA_PATH=""    # .../commonsense_170k_taskid.json
CACHE_DIR=""    # Cache directory is not used in this script
DEEPSPEED_CONFIG=config/ds2.json
OUTPUT_PATH=""  # Output directory is not used in this script

unset WANDB_RUN_ID
unset WANDB_RUN_NAME

deepspeed \
    --include="node-0" \
    --master_port=25000 \
    $SCRIPT_PATH \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --batch_size 8  \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \
    --save_step 1000  \
    --adapter_name mlora \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_r $1 \
    --lora_alpha $2 \
    --use_gradient_checkpointing \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --cache_dir $CACHE_DIR \
    --deepspeed $DEEPSPEED_CONFIG \