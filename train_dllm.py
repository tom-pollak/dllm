#!/usr/bin/env -S uv run torchrun --standalone --nproc_per_node=gpu
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate",
#   "datasets",
#   "torch",
#   "transformers",
#   "wandb",
# ]
# ///

# dice = random.random()
# ids[i] = (
#     mask_id if dice < .1
#     else random.randint(0, tok.vocab_size - 1) if dice < .9
#     else ids[i]
# )

import os, random, torch
from datasets import load_dataset
from accelerate import notebook_launcher
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
def main():
    assert os.environ.get("HF_TOKEN") and os.environ.get("WANDB_API_KEY")
    model_id = "answerdotai/ModernBERT-large"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    mask_id, sep_id, sep = tok.mask_token_id, tok.sep_token_id, tok.sep_token
    tok.chat_template = (
        "User: {{ messages[0]['content'] }}\n" + sep + "\nAssistant:\n{{ messages[1]['content'] }}")
    ## Dataset
    def preprocess(batch):
        def _single(messages):
            text = tok.apply_chat_template(messages, tokenize=False)
            enc = tok(text, truncation=True, padding="max_length", max_length=512)
            ids, labels = enc["input_ids"], [-100] * len(enc["input_ids"])
            if sep_id not in ids: return None
            start = ids.index(sep_id) + 1
            cand = [i for i in range(start, len(ids))
                    if ids[i] not in (tok.pad_token_id, sep_id)]
            if not cand: return None
            n_mask = max(int(len(cand) * random.uniform(0.15, 0.99)), 1)
            for i in random.sample(cand, n_mask):
                labels[i] = ids[i]
                ids[i] = mask_id
            return ids, labels

        mapped = (_single(m) for m in batch["messages"])
        filtered = (tup for tup in mapped if tup is not None)
        ids, labels = zip(*filtered)
        return {"input_ids": list(ids), "labels": list(labels)}

    ds = load_dataset("allenai/tulu-3-sft-mixture-0225", split="train")
    dd = (ds
        .map(preprocess, num_proc=32, batched=True, remove_columns=ds.column_names)
        .train_test_split(0.05, seed=42))

    ## Train
    project_name, run_name = "dllm", "modernbert-flowlm-tulu"
    os.environ.setdefault("WANDB_PROJECT", project_name)
    args = TrainingArguments(
        run_name, num_train_epochs=1,
        per_device_train_batch_size=32, per_device_eval_batch_size=32,
        lr_scheduler_type="cosine", bf16=True,
        eval_strategy="steps", eval_steps=200,
        report_to="wandb", push_to_hub=True,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=dd["train"], eval_dataset=dd["test"],
    )
    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__": main()
# notebook_launcher(main, num_processes=torch.cuda.device_count())
