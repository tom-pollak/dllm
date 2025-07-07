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
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ids = inputs["input_ids"]                      # (B,S)
        device = ids.device
        B, S = ids.shape
        hot = F.one_hot(ids, V).float()

        with self.accelerator.autocast():
            t = torch.rand(B, 1, 1).to(device) * 0.12 + 0.03
            alpha = 1.0 - t

            eps = torch.randn_like(hot)
            w_t = alpha * hot + torch.sqrt(1.0 - alpha ** 2) * eps

            step = self.state.global_step
            temp = max(1e-3 * (1 - step / 5e5), 0.0) + 1e-8
            probs = torch.softmax(w_t / temp, -1).to(torch.bfloat16)
            W_E = model.get_input_embeddings().weight
            emb   = probs @ W_E
            attn = (ids != pad_token_id).int()
            out   = model(inputs_embeds=emb, attention_mask=attn)
            logp  = torch.log_softmax(out.logits, -1)


        kl_qp = F.kl_div(logp, hot, reduction="none")
        logq = (hot + 1e-8).log()
        kl_pq = F.kl_div(logq, logp.exp(), reduction="none", log_target=True)
        kl = (kl_qp + kl_pq).sum(-1)

        scale = (1.0 - alpha.squeeze(-1).squeeze(-1)) / (alpha.squeeze(-1).squeeze(-1) + 1e-8)
        mask = (ids != pad_token_id).float()
        loss = (kl * scale[:, None] * mask).sum() / mask.sum()

        return (loss, out) if return_outputs else loss

run_name = "modernbert-duo-tulu"
os.environ.setdefault("WANDB_PROJECT", run_name)
args = TrainingArguments(
    run_name, bf16=True,
    per_device_train_batch_size=32, per_device_eval_batch_size=32,
    eval_strategy="steps", eval_steps=1000, num_train_epochs=1,
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
tok.push_to_hub(f"tommyp111/{run_name}")

