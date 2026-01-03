# train_grpo.py
import argparse
import os

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
from math_verify import parse


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    return model, tokenizer


def train(args: argparse.Namespace):
    # Configure logging
    if args.log:
        os.environ["WANDB_PROJECT"] = "test-grpo"
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # Load dataset
    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")
    dataset = dataset.filter(lambda x: len(parse(x["solution"])) > 0, num_proc=4)

    # Load model
    model, tokenizer = load_model("Qwen/Qwen2-0.5B-Instruct")

    # Instantiate trainer
    training_args = GRPOConfig(
        report_to="wandb" if args.log else "none",
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--use_vllm", action="store_true", help="Use vLLM for generation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
