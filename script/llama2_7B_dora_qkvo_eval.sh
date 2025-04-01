CUDA_VISIBLE_DEVICES=0 python lora_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset boolq \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/boolq.txt \
    &

CUDA_VISIBLE_DEVICES=1 python lora_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset piqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/piqa.txt \
    &

CUDA_VISIBLE_DEVICES=2 python lora_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset social_i_qa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/social_i_qa.txt \
    &

CUDA_VISIBLE_DEVICES=3 python lora_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset winogrande \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/winogrande.txt \
    &

CUDA_VISIBLE_DEVICES=4 python lora_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset openbookqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/openbookqa.txt \
    &

CUDA_VISIBLE_DEVICES=5 python lora_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset hellaswag \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/hellaswag.txt \
    &

CUDA_VISIBLE_DEVICES=6 python lora_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset ARC-Challenge \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/arc-c.txt \
    &

CUDA_VISIBLE_DEVICES=7 python lora_evaluate.py \
    --model LLaMA-7B \
    --adapter DoRA \
    --dataset ARC-Easy \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --batch_size 1 \
    --lora_weights $1|tee -a $2/arc-e.txt \
    &

wait

echo "Done"