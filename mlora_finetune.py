import ast
import os
import sys
from functools import partial
from typing import List

import torch
import transformers
from datasets import load_dataset, load_from_disk
from deepspeed.utils.logging import LoggerFactory
from src.custom_model import LlamaForCausalLM
from src.utils import add_filehandler, save_pretrain, set_no_grad, wrap_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

logger = LoggerFactory.create_logger(__name__)
sys.path.append(os.path.join(os.getcwd(), "~/MTL-LoRA"))

# task_name_to_id = {
#     "boolq": 0,
#     "piqa": 1,
#     "social_i_qa": 2,
#     "hellaswag": 3,
#     "winogrande": 4,
#     "ARC-Challenge": 5,
#     "ARC-Easy": 6,
#     "openbookqa": 7,
# }


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""


import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="base model to use for finetuning",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="cache tokenized data",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="yahma/alpaca-cleaned",
        help="path to dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora-alpaca",
        help="output directory",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="lora",
        help="adapter type to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay",
    )
    parser.add_argument(
        "--cutoff_len",
        type=int,
        default=256,
        help="max sequence length",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="use gradient checkpointing",
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=200,
        help="save step",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="lora r",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="lora alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="lora dropout",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=ast.literal_eval,
        default=None,
        help="lora target modules",
    )
    # mlora hyperparams
    parser.add_argument(
        "--lambda_num",
        type=int,
        default=3,
        help="lambda num",
    )
    parser.add_argument(
        "--num_B",
        type=int,
        default=3,
        help="num B",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature",
    )
    # multilora hyperparams
    parser.add_argument(
        "--lora_num",
        type=int,
        default=3,
        help="lora num",
    )
    # moelora hyperparams
    parser.add_argument(
        "--expert_num",
        type=int,
        default=3,
        help="expert num",
    )
    parser.add_argument(
        "--task_num",
        type=int,
        default=8,
        help="task num",
    )
    parser.add_argument(
        "--te_dim",
        type=int,
        default=64,
        help="te dim",
    )
    # dora hyperparams
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=False,
        help="merge weights",
    )
    parser.add_argument(
        "--Wdecompose",
        type=bool,
        default=False,
        help="Wdecompose",
    )
    parser.add_argument(
        "--dora_simple",
        type=bool,
        default=True,
        help="dora simple",
    )
    # misc
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="wandb project",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="wandb run name",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="load from checkpoint",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="",
        help="deepspeed",
    )
    parser.add_argument(
        "--train_on_inputs",
        type=bool,
        default=False,
        help="train on inputs",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
    )

    return parser.parse_args()


def train(
    base_model: str = "",
    cache_dir: str = None,  # cache tokenized data
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    adapter_name: str = "lora",
    batch_size: int = 128,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    cutoff_len: int = 256,
    use_gradient_checkpointing: bool = False,
    save_step: int = 200,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    # mlora hyperparams
    lambda_num: int = 3,
    num_B: int = 3,
    temperature: float = 1.0,
    # multilora hyperparams
    lora_num: int = 3,
    # moelora hyperparams
    expert_num: int = 3,
    task_num: int = 8,
    te_dim: int = 64,
    # misc
    gradient_accumulation_steps: int = 1,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter (resume training)
    load_from_checkpoint: str = None,  # either training checkpoint or final adapter
    deepspeed: str = "",
    train_on_inputs: bool = False,
    merge_weights: bool = False,
    Wdecompose: bool = False,
    dora_simple: bool = True,
    **kwargs,
):
    use_wandb = wandb_project is not None
    add_filehandler(logger, os.path.join(output_dir, "logging"))
    if "llama" in base_model and adapter_name.lower() in ["mlora", "moelora"]:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if model.config.model_type == "llama":
        if "Llama-3" in base_model:
            logger.info("load llama-3 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.enable_input_require_grads()

    if adapter_name.lower() == "mlora":
        mlora_config = {
            "type": "mlora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lambda_num": lambda_num,
            "B_num": num_B,
            "B_scale": temperature,
            "diagonal_format": False,
        }
    elif adapter_name.lower() == "multilora":
        mlora_config = {
            "type": "multilora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_num": lora_num,
        }
    elif adapter_name.lower() == "moelora":
        mlora_config = {
            "type": "moelora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "expert_num": expert_num,
            "task_num": task_num,
            "task_embedding_dim": te_dim,
        }
    elif adapter_name.lower() == "dora":
        mlora_config = {
            "type": "dora",
            "r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "merge_weights": merge_weights,
            "Wdecompose": Wdecompose,
            "dora_simple": dora_simple,
        }

    model = wrap_model(model, lora_target_modules, mlora_config)
    if load_from_checkpoint is not None:
        state_dict = torch.load(load_from_checkpoint, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(msg.unexpected_keys)
    set_no_grad(model, logger=logger)
    model.config.use_cache = False

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point, debug_tokenization=False):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably

        if adapter_name.lower() in ["mlora", "moelora"]:
            tokenized_full_prompt["lambda_index"] = data_point["task_id"]
        return tokenized_full_prompt

    if cache_dir is not None and os.path.exists(os.path.join(cache_dir, "train")):
        train_data = load_from_disk(os.path.join(cache_dir, "train"))
    else:
        if data_path.endswith(".json"):  # todo: support jsonl
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)

        train_data = (
            data["train"]
            .shuffle()
            .map(
                partial(generate_and_tokenize_prompt, debug_tokenization=tokenizer),
                batched=False,
                desc="Tokenizing train data",
                num_proc=4,
            )
        )

        if cache_dir is not None:
            train_data.save_to_disk(os.path.join(cache_dir, "train"))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.01,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            bf16=True,
            deepspeed=deepspeed,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="no",
            save_only_model=True,
            eval_steps=None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            report_to=(["wandb", "tensorboard"] if use_wandb else ["tensorboard"]),
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", padding=True, max_length=cutoff_len
        ),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    save_pretrain(model, output_dir, prefix=["lora"])


if __name__ == "__main__":
    args = arg_parser()
    train(
        **vars(args),
    )
