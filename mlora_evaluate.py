import argparse
import ast
import copy
import json
import os
import re
import sys

import torch
from src.custom_model import LlamaForCausalLM
from src.utils import wrap_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
)

sys.path.append(os.path.join(os.getcwd(), "~/MTL-LoRA"))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


task_name_to_id = {
    "boolq": 0,
    "piqa": 1,
    "social_i_qa": 2,
    "hellaswag": 3,
    "winogrande": 4,
    "ARC-Challenge": 5,
    "ARC-Easy": 6,
    "openbookqa": 7,
}


def main():
    args = parse_args()

    def evaluate(
        instructions,
        input=None,
        temperature=1.0,
        top_p=1,
        top_k=1,
        num_beams=1,
        max_new_tokens=32,
        **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)

        if args.adapter in ["mlora", "moelora"]:
            lambda_index = task_name_to_id[args.dataset]
            lambda_index = (
                torch.tensor(lambda_index).repeat(input_ids.shape[0]).to(device)
            )
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=False,
            **kwargs,
        )
        with torch.no_grad():
            if args.adapter in ["mlora", "moelora"]:
                generation_output = model.generate(
                    input_ids=input_ids,
                    lambda_index=lambda_index,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                )
            else:
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        return outputs

    save_file = f"experiment/{args.model}-{args.adapter}-{args.dataset}.json"
    os.makedirs("experiment", exist_ok=True)

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    model.to(torch.bfloat16)
    model.eval()
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get("instruction") for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get("answer")
            flag = False
            predict = extract_answer(args, output)
            if label == predict or (predict in label and predict != ""):
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = predict
            new_data["flag"] = flag
            output_data.append(new_data)
            print(data["instruction"])
            print(output)
            print("prediction:", predict)
            print("label:", label)
        print("---------------")
        print(f"\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}")
        print("---------------")
        with open(save_file, "w+") as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print("\n")
    print("test finished")


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f"/home/aiscuser/mLoRA/dataset/{args.dataset}/test.json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, "r"))
    return json_data


def create_batch(dataset, batch_size):
    batches = []
    num_batch = (
        len(dataset) // batch_size
        if len(dataset) % batch_size == 0
        else len(dataset) // batch_size + 1
    )
    for i in range(num_batch):
        batch = dataset[i * batch_size : min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "boolq",
            "piqa",
            "social_i_qa",
            "hellaswag",
            "winogrande",
            "ARC-Challenge",
            "ARC-Easy",
            "openbookqa",
        ],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["LLaMA-7B", "LLaMA-13B", "BLOOM-7B", "GPT-j-6B"],
        required=True,
    )
    parser.add_argument(
        "--adapter",
        choices=["mlora", "moelora", "multilora", "dora"],
        required=True,
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_weights", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lora_target_modules", type=ast.literal_eval, required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # mlora
    parser.add_argument("--lambda_num", type=int)
    parser.add_argument("--num_B", type=int)
    parser.add_argument("--temperature", type=float)
    # multilora
    parser.add_argument("--lora_num", type=int)
    # moelora
    parser.add_argument("--expert_num", type=int)
    parser.add_argument("--task_num", type=int)
    parser.add_argument("--te_dim", type=int)
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

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f"can not find base model name by the value: {args.model}")
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f"can not find lora weight, the value is: {lora_weights}")

    if "LLaMA" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if "llama" in base_model.lower() and args.adapter.lower() in [
        "mlora",
        "moelora",
    ]:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )  # fix zwq

    if args.adapter.lower() == "mlora":
        mlora_config = {
            "type": "mlora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lambda_num": args.lambda_num,
            "B_num": args.num_B,
            "B_scale": args.temperature,
            "diagonal_format": False,
        }
    elif args.adapter.lower() == "multilora":
        mlora_config = {
            "type": "multilora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_num": args.lora_num,
        }
    elif args.adapter.lower() == "moelora":
        mlora_config = {
            "type": "moelora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "expert_num": args.expert_num,
            "task_num": args.task_num,
            "task_embedding_dim": args.te_dim,
        }
    elif args.adapter.lower() == "dora":
        mlora_config = {
            "type": "dora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "merge_weights": args.merge_weights,
            "Wdecompose": args.Wdecompose,
            "dora_simple": args.dora_simple,
        }

    model = wrap_model(model, args.lora_target_modules, mlora_config)

    state_dict = torch.load(lora_weights, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg.unexpected_keys)
    assert len(msg.unexpected_keys) == 0

    for name, param in model.named_parameters():
        param.requires_grad = False

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ""
    if not instruction:
        raise ValueError("instruct not initialized")
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == "boolq":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"true|false", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "piqa":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"solution1|solution2", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
        if not pred_answers:
            pred_answers = re.findall(r"1|2|3|4", sentence_)
            if not pred_answers:
                return ""
        return pred_answers[0]
    elif dataset == "hellaswag":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "winogrande":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"option1|option2", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()
