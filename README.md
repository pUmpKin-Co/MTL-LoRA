<h1 align="center"> MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning </h1> 

<p align="center">
  <a href="https://arxiv.org/abs/2410.09437"><img src="https://img.shields.io/badge/arXiv-2402.13949-b31b1b.svg" alt="arXiv"></a>
</p>

<h5 align="center"><em>Yaming Yang*, Dilxat Muhtar*, Yelong Shen* , Yuefeng Zhan, Jianfeng Liu, Yujing Wang, Hao Sun, Denvy Deng, Feng Sun, Qi Zhang, Weizhu Chen, and Yunhai tong</em>
</br>(*Equal Contribution)</h5>

## Introduction
**MTL-LoRA** enhances the multi-task learning capabilities of LoRA by introducing task-adaptive parameters that distinguish task-specific information while efficiently capturing shared knowledge across tasks in low-dimensional spaces. MTL-LoRA outperforms LoRA and its variants with comparable or fewer learnable parameters in multi-task learning scenarios.

## Usage and Reproducing
### Implementation
The standalone implementation of MTL-LoRA can be found [here](./src/adapter/mlora.py). Currently, we only support for tuning linear layer.

### Commonsense Reasoning
+ Setup
    ```
    conda create -n mtl-lora python==3.10 -y
    conda activate mtl-lora
    pip install torch
    pip install -r requirements.txt
    ```
+ Datasets

    + Please follow the instruction [here](https://github.com/NVlabs/DoRA/tree/main/commonsense_reasoning) to prepare the dataset.
    + For training MTL-LoRA or MoE-LoRA, please use the dataset with task IDs available [here](./commonsense_170k_taskid.json)
+ Fine-tuning
    + The script for fine-tuning different adapters can be found at `./script`.
    + For fine-tuning with MTL-LoRA:

        + Specify the `DATA_PATH`, `OUTPUT_PATH`, and `CACHE_PATH` (For caching the tokenized data).
        ```
        bash ./script/llama2_7B_mlora_qkvo.sh 8 16
        ```
+ Evaluation
  
    + The script for evaluation can be found at `./script`
    + For evaluation MTL-LoRA:
        ```
        bash ./script/llama2_7B_mlora_qkvo_eval.sh $CHECKPOINT_PATH $OUTPUT_PATH
        ```

### Image-Text Undetstanding
Please follow the instruction [here](./image_text_understanding/README.md).

## Acknowledgements
We gratitude to the following repositories for their wonderful works:
+ [LoRA](https://github.com/microsoft/LoRA)
+ [DoRA](https://github.com/NVlabs/DoRA)

## Contact

Please contact us or post an issue if you have any questions: pumpkindilxat@gmail.com.