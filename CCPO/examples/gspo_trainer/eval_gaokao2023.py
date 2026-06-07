"""
Standalone evaluation script for Gaokao2023 math dataset.
Uses vLLM directly for generation (bypasses verl's async-only rollout).
"""

import argparse
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="HuggingFace model directory")
    p.add_argument("--data_path", required=True, help="Input parquet with 'prompt' column")
    p.add_argument("--output_path", required=True, help="Output parquet with 'responses' column")
    p.add_argument("--prompt_key", default="prompt")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--pipeline_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_response_length", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--skip_eval", action="store_true", help="Skip accuracy evaluation")
    return p.parse_args()


def generate(args):
    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from {args.data_path} ...")
    dataset = pd.read_parquet(args.data_path)
    chat_lst = dataset[args.prompt_key].tolist()
    chat_lst = [c.tolist() if hasattr(c, "tolist") else c for c in chat_lst]

    print(f"Applying chat template to {len(chat_lst)} prompts ...")
    prompt_texts = [
        tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        for chat in chat_lst
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_response_length,
        n=args.n_samples,
    )

    print(f"Loading vLLM model from {args.model_path} ...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_length + args.max_response_length,
        trust_remote_code=args.trust_remote_code,
        dtype="bfloat16",
    )

    print("Generating responses ...")
    outputs = llm.generate(prompt_texts, sampling_params)

    responses_per_sample = []
    for out in outputs:
        responses_per_sample.append([o.text for o in out.outputs])

    dataset["responses"] = responses_per_sample
    dataset.to_parquet(args.output_path)
    print(f"Results saved to {args.output_path}")
    return dataset


def evaluate(df):
    sys.path.insert(0, "/gz-data/verl-gspo-VD-double")
    from verl.utils.reward_score import default_compute_score

    total, correct = 0, 0
    for idx, row in df.iterrows():
        data_source = row.get("data_source", "ReMA-math")
        responses = row["responses"]
        if isinstance(responses, np.ndarray):
            responses = responses.tolist()
        if not responses:
            continue

        response = responses[0]
        reward_model = row["reward_model"]
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth")
        else:
            ground_truth = row.get("groundtruth")
        if not ground_truth:
            continue

        score = default_compute_score(data_source, response, ground_truth)
        val = float(score) if isinstance(score, (int, float)) else float(score.get("score", 0.0))
        if val == 1.0:
            correct += 1
        total += 1

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)}, current accuracy: {correct/total:.4f}")

    if total > 0:
        print(f"Final Accuracy: {correct/total:.4f} ({correct}/{total})")
    else:
        print("No samples evaluated.")


def main():
    args = parse_args()
    df = generate(args)
    if not args.skip_eval:
        evaluate(df)


if __name__ == "__main__":
    main()
