CUDA_VISIBLE_DEVICES=$2 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter multilora \
    --dataset boolq \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_r $3 \
    --lora_alpha $4 \
    --lora_num 3 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_weights $1|tee -a $5/boolq.txt

CUDA_VISIBLE_DEVICES=$2 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter multilora \
    --dataset piqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_r $3 \
    --lora_alpha $4 \
    --lora_num 3 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_weights $1|tee -a $5/piqa.txt

CUDA_VISIBLE_DEVICES=$2 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter multilora \
    --dataset social_i_qa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_r $3 \
    --lora_alpha $4 \
    --lora_num 3 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_weights $1|tee -a $5/social_i_qa.txt

CUDA_VISIBLE_DEVICES=$2 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter multilora \
    --dataset hellaswag \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_r $3 \
    --lora_alpha $4 \
    --lora_num 3 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_weights $1|tee -a $5/hellaswag.txt

CUDA_VISIBLE_DEVICES=$2 python ~/MTL-LoRA/mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter multilora \
    --dataset winogrande \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_r $3 \
    --lora_alpha $4 \
    --lora_num 3 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_weights $1|tee -a $5/winogrande.txt

CUDA_VISIBLE_DEVICES=$2 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter multilora \
    --dataset ARC-Challenge \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_r $3 \
    --lora_alpha $4 \
    --lora_num 3 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_weights $1|tee -a $5/ARC-Challenge.txt

CUDA_VISIBLE_DEVICES=$2 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter multilora \
    --dataset ARC-Easy \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_r $3 \
    --lora_alpha $4 \
    --lora_num 3 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_weights $1|tee -a $5/ARC-Easy.txt

CUDA_VISIBLE_DEVICES=$2 python mlora_evaluate.py \
    --model LLaMA-7B \
    --adapter multilora \
    --dataset openbookqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_r $3 \
    --lora_alpha $4 \
    --lora_num 3 \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_weights $1|tee -a $5/openbookqa.txt