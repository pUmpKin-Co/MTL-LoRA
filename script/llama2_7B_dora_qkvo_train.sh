SCRIPT_PATH=lora_finetune.py
DATA_PATH=""    # ../commonsense_170k.json
DEEPSPEED_CONFIG=config/ds2.json
OUTPUT_PATH=""  # Output directory is not used in this script

unset WANDB_RUN_ID
unset WANDB_RUN_NAME

deepspeed $SCRIPT_PATH \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --batch_size 8 \
    --num_epochs 3 \
    --learning_rate 3e-4 --cutoff_len 512 \
    --save_step 1000  --adapter_name dora \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_r $1 --lora_alpha $2 --use_gradient_checkpointing \
    --deepspeed $DEEPSPEED_CONFIG \
