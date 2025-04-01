task=multitask

model="bart"

echo $model
export CUDA_VISIBLE_DEVICES=2

if [ $model == "t5" ]
then
    folder_prefix="VLT5"
    backbone="t5-base"
    batch_size=400
elif [ $model == "bart" ]
then
    folder_prefix="VLBart"
    backbone="facebook/bart-base"
    batch_size=512
fi

echo $folder_prefix
echo $backbone

feature=RN101

lr=1e-3

lora_dim=64

num_B=2

mlora_temperature=0.8

lora_alpha=16

unset WANDB_RUN_ID
unset WANDB_RUN_NAME

project_name=mLoRA
run_name=vl_lr${lr}_r${lora_dim}_B${num_B}_alpha${lora_alpha}_t${mlora_temperature}_wr03_bs${batch_size}
output=~/Output/${folder_prefix}_${task}/$run_name
# vqa: 0
# gqa: 1
# nlvr: 2
# caption: 3
TOKENIZERS_PARALLELISM=True PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_port=29500 \
    ./VL-T5/src/${task}.py \
    --distributed \
    --multiGPU \
    --optim adamw \
    --warmup_ratio 0.03 \
    --clip_grad_norm 5 \
    --lr ${lr} \
    --epochs 20 \
    --num_workers 4 \
    --backbone ${backbone} \
    --output $output ${@:2} \
    --num_beams 35 \
    --use_tasks_prompts \
    --batch_size ${batch_size} \
    --valid_batch_size ${batch_size} \
    --use_mlora \
    --B_num ${num_B} \
    --lora_alpha ${lora_alpha} \
    --mlora_temperature ${mlora_temperature} \
    --lambda_num 4 \
    --lora_settings \
    --lora_dim ${lora_dim} \
    --tasks "vqa,gqa,nlvr,caption" \
    --feature ${feature} \
    --n_boxes 36 \
    --downsample \
    --image_size "(224,224)" \
    --project_name $project_name \
    --run_name $run_name \
    --unfreeze_bias \
    --unfreeze_layer_norms \