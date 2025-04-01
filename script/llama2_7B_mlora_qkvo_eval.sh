CUDA_VISIBLE_DEVICES=0 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset boolq \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $1|tee -a $2/boolq.txt \
    &

CUDA_VISIBLE_DEVICES=1 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset piqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $1|tee -a $2/piqa.txt \
    &

CUDA_VISIBLE_DEVICES=2 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset social_i_qa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $1|tee -a $2/social_i_qa.txt \
    &

CUDA_VISIBLE_DEVICES=3 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset hellaswag \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $1|tee -a $2/hellaswag.txt \
    &

CUDA_VISIBLE_DEVICES=4 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset winogrande \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $1|tee -a $2/winogrande.txt \
    &

CUDA_VISIBLE_DEVICES=5 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset ARC-Challenge \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $1|tee -a $2/ARC-Challenge.txt \
    &

CUDA_VISIBLE_DEVICES=6 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset ARC-Easy \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $1|tee -a $2/ARC-Easy.txt \
    &

CUDA_VISIBLE_DEVICES=7 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset openbookqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $1|tee -a $2/openbookqa.txt \
    &

wait

echo "Done"
