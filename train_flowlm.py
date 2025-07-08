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
import os
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          DataCollatorWithPadding, TrainingArguments, Trainer)

model_id = "answerdotai/ModernBERT-large"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, reference_compile=False)

V = model.config.vocab_size
pad_token_id, sep_id, sep = tok.pad_token_id, tok.sep_token_id, tok.sep_token
tok.chat_template = (
    "User: {{ messages[0]['content'] }}\n" + sep +
    "\nAssistant:\n{% generation %}{{ messages[1]['content'] }}{% endgeneration %}")


## Dataset
def preprocess(batch):
    def _one(messages):
        txt = tok.apply_chat_template(messages, tokenize=False)
        enc = tok(txt, truncation=True, padding="max_length", max_length=512)
        ids = enc["input_ids"]
        return ids if sep_id in ids else None
    rows = [_one(m) for m in batch["messages"]]
    rows = [r for r in rows if r is not None]
    return {"input_ids": rows}

ds = load_dataset("allenai/tulu-3-sft-mixture-0225", split="train")
dd = (ds
      .map(preprocess, batched=True, remove_columns=ds.column_names, num_proc=32)
      .train_test_split(0.05, seed=42))

## Train
class DuoTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        ids   = inputs["input_ids"]  # (B,S)
        B, S  = ids.shape
        device = ids.device

        hot = F.one_hot(ids, V).float()  # (B,S,V)

        # Duo Corruption
        t      = torch.rand((B,), device=device) * 0.12 + 0.03   # blur level
        alpha  = 1.0 - t[:, None, None]                          # signal mix
        eps    = torch.randn_like(hot)                           # N(0,1) noise
        w_t    = alpha * hot + torch.sqrt(1.0 - alpha**2) * eps  # noisy logits

        step  = self.state.global_step
        temperature = max(1e-3 * (1 - step / 5e5), 0.0) + 1e-8  # τ-anneal
        probs = torch.softmax(w_t / temperature, dim=-1)

        # Map back to embedding space
        W_E = model.get_input_embeddings().weight
        emb   = probs.to(W_E.dtype) @ W_E  # (B,S,D)
        attn  = (ids != pad_token_id).int()

        # Forward pass
        with self.accelerator.autocast():
            logits = model(inputs_embeds=emb, attention_mask=attn).logits  # (B,S,V)
            logp   = F.log_softmax(logits, dim=-1)

        # Loss
        kl_pq = F.kl_div(logp, hot, reduction='none')     # KL(p||q)
        kl_qp = F.kl_div((hot + 1e-8).log(), logp.exp(),  # KL(q||p)
                         reduction='none', log_target=True)
        sym_kl = (kl_pq + kl_qp).sum(-1)  # (B,S)

        view_scale = t / (V * (1 - t) + t)  # (B,)

        mask  = (ids != pad_token_id).float()
        loss  = (view_scale[:, None] * sym_kl * mask).sum() / mask.sum()

        return (loss, logits) if return_outputs else loss

project_name = "modernbert-duo-tulu"
os.environ.setdefault("WANDB_PROJECT", project_name)
args = TrainingArguments(
    bf16=True,
    per_device_train_batch_size=32, per_device_eval_batch_size=32,
    num_train_epochs=1, logging_steps=20, eval_strategy="steps", eval_steps=1000,
    report_to="wandb", push_to_hub=True,
    deepspeed={
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "bf16": {"enabled": True},
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    },
)

trainer = DuoTrainer(
    model=model, args=args,
    train_dataset=dd["train"], eval_dataset=dd["test"],
    data_collator=DataCollatorWithPadding(tok),
)
trainer.train()
tok.push_to_hub(f"tommyp111/{project_name}")
