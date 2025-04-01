import argparse
import ast
import os
from typing import List

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict  # noqa: E402
from transformers import (  # noqa: F402
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="The base model to use for training",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="yahma/alpaca-cleaned",
        help="The path to the dataset to use for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora-alpaca",
        help="The directory to save the trained model",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="lora",
        help="The adapter to use for training",
    )
    parser.add_argument(
        "--load_8bit",
        type=bool,
        default=False,
        help="Whether to load the model in 8-bit",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="The number of epochs to train for",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="The learning rate for training",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="The weight decay for training",
    )
    parser.add_argument(
        "--cutoff_len",
        type=int,
        default=256,
        help="The cutoff length for the input",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing",
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=200,
        help="The number of steps to save the model",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="The number of heads for LoRA",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The alpha value for LoRA",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="The dropout value for LoRA",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=ast.literal_eval,
        default=None,
        help="The target modules for LoRA",
    )
    parser.add_argument(
        "--train_on_inputs",
        type=bool,
        default=False,
        help="Whether to train on inputs",
    )
    parser.add_argument(
        "--group_by_length",
        type=bool,
        default=False,
        help="Whether to group by length",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="The project to use for wandb",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="The run name to use for wandb",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="The checkpoint to resume from",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="",
        help="The deepspeed configuration to use",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="The local rank for distributed training",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="task_id",
    )

    return parser.parse_args()


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    adapter_name: str = "lora",
    load_8bit: bool = False,
    # training hyperparams
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
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = None,
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    deepspeed: str = "",
    task_id: int = 0,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    use_wandb = wandb_project is not None

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )

    if model.config.model_type == "llama":
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        # need to handle llama 3 separately
        if "Llama-3" in base_model:
            print("load llama-3 tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
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
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {
                "input_ids": result["input_ids"],
                "labels": result["labels"],
            }
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
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
        return tokenized_full_prompt

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    print(model)
    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, config)
    if adapter_name == "prefix-tuning":
        model.to("cuda")

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data = data.filter(lambda x: x["task_id"] == task_id)
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            warmup_ratio=0.03,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            bf16=True,
            deepspeed=deepspeed,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=save_step,
            output_dir=output_dir,
            save_safetensors=False,
            save_total_limit=3,
            save_only_model=True,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else "tensorboard",
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir, safe_serialization=False)


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501


if __name__ == "__main__":
    arg = argparser()
    train(
        base_model=arg.base_model,
        data_path=arg.data_path,
        output_dir=arg.output_dir,
        adapter_name=arg.adapter_name,
        load_8bit=arg.load_8bit,
        batch_size=arg.batch_size,
        num_epochs=arg.num_epochs,
        learning_rate=arg.learning_rate,
        weight_decay=arg.weight_decay,
        cutoff_len=arg.cutoff_len,
        use_gradient_checkpointing=arg.use_gradient_checkpointing,
        save_step=arg.save_step,
        lora_r=arg.lora_r,
        lora_alpha=arg.lora_alpha,
        lora_dropout=arg.lora_dropout,
        lora_target_modules=arg.lora_target_modules,
        train_on_inputs=arg.train_on_inputs,
        group_by_length=arg.group_by_length,
        wandb_project=arg.wandb_project,
        wandb_run_name=arg.wandb_run_name,
        resume_from_checkpoint=arg.resume_from_checkpoint,
        deepspeed=arg.deepspeed,
        task_id=arg.task_id,
    )
