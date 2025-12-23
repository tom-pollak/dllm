# Diffusion Language Models

## What are Language Diffusion Models?

Instead of predicting the next token, a diffusion LLM must "de-mask" a corrupted sequence of text.

# LLaDA: Large Language Diffusion with mAsking

The paper introduces **masked diffusion models**.

These generate text by **iteratively denoising** a corrupted sequence.

```
[MASK] [MASK]  [MASK] [MASK] [MASK] [MASK] [EOS]
[MASK] [MASK]  [MASK] France [MASK] [MASK] [EOS]
The    [MASK]  [MASK] France [MASK] [MASK] [EOS]
The    capital [MASK] France [MASK] [MASK] [EOS]
The    capital [MASK] France is     [MASK] [EOS]
The    capital [MASK] France is     Paris  [EOS]
The    capital of     France is     Paris  [EOS]
```

In inference, this allows the model to produce multiple tokens per step.

## Training

> **The core idea is very similar to BERT**
>
> My minimal training setup at: [train_dllm.py](./train_dllm.py). This just takes a pretrained BERT model and finetunes with a variable amount of masking.
>
> Run with `./train_dllm.py` (8 H100s).

We corrupt part of the input sequence with `[MASK]` tokens, and train the model to predict these tokens using cross-entropy loss (CEL).

However, while BERT is trained with a **fixed** 15% mask rate, LLaDA uses variable 15-99% masking.

A disadvantage of BERT-like models: only get training signal from masked tokens.

```
The [MASK] of France [MASK] Paris [EOS]
       ^                ^
  Only the masked tokens contribute to the loss
```

### A Minimal Inference Setup



```python
import sys
import time
import torch

@torch.no_grad()
def iter_mask_decode(model, tokenizer, prompt: str, answer_length: int = 32):
    # Create initial sequence with masked tokens
    assistant_message = tokenizer.mask_token * answer_length
    toks_dict = tokenizer.apply_chat_template(
        [{"content": prompt}, {"content": assistant_message}],
        return_dict=True, return_assistant_tokens_mask=True, return_tensors="pt")
    
    ids = toks_dict['input_ids'][0].tolist()
    assistant_mask = toks_dict['assistant_masks']
    answer_start = assistant_mask[0].nonzero().min().item()
    
    device = next(model.parameters()).device
    
    for step in range(answer_length):
        logits = model(input_ids=torch.tensor([ids]).to(device)).logits
        probs = torch.softmax(logits[0], dim=-1)

        mask_positions = (torch.tensor(ids) == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            break

        mask_probs = probs[mask_positions]
        confidence_scores = mask_probs.max(dim=-1)[0]
        best_idx = confidence_scores.argmax()

        pos = mask_positions[best_idx]
        new_token = mask_probs[best_idx].argmax().item()
        ids[pos] = new_token
        
        yield new_token, pos.item() - answer_start

def demo_inference(model, tokenizer, prompt: str, answer_length: int = 25, delay=0.1):
    def _print_step(resp, n_clear):
        """1) move to start, 2) blank the full width, 3) move back, 4) write new text"""
        resp = resp.encode('unicode_escape').decode('ascii')
        blank = " " * n_clear
        sys.stdout.write("\r" + blank + "\r" + resp)
        sys.stdout.flush()
        return len(resp)
    
    print(f"User: {prompt}")
    print("Assistant: ", end="")
    
    tokens = ["[MASK]"] * answer_length
    n_clear = 0
    
    for new_token, pos in iter_mask_decode(model, tokenizer, prompt, answer_length):
        tokens[pos] = tokenizer.decode(new_token)
        resp = "".join(tokens)
        n_clear = _print_step(resp, n_clear)
        time.sleep(delay)
    
    print()
```


```python
import os, random, itertools, math, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

device =  (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

model_id = "tommyp111/modernbert-dllm-tulu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()

prompt = "What is the meaning of life?"
```

## Inference

> "Greedy strategy"

Iteratively "de-mask" the most confident token at the most confidence point in the sequence.


```python
demo_inference(model, tokenizer, prompt, answer_length=25, delay=0.25)
```

    User: What is the meaning of life?
    \nThe meaning of life is to live, to love, to laugh, to cry, to dream, to hope,[SEP]                                                              


### Block Generation with KV cache.

Key limitation: **KV caching is not possible**, since at the newly de-masked token attention is all-to-all.

Mitigated by generating "blocks" of tokens at a time, autoregressively.

E.g: To generate 128 tokens, we autoregressively generate block 4 blocks of 32.

### Remasking

A key limitation of masked diffusion LMs is they have no **self-correction**

The authors suggest confidence based remasking.

# Diffusion Duality (Duo)

Primary issue with masked diffusion models: No ability for self correction.

For inference: Model starts from **random tokens**, that will be iteratively refined.

- This allows it to self-correct it's previous predictions!

> See training script: [train_flowlm.py](./train_flowlm.py)

## Duo's Approach:

1. Corrupt one-hot tokens with a variable amount of gaussian noise.
2. Convert back to one-hot (with argmax), some tokens will have flipped. This is our **corrupted sequence**.
3. Train model to restore original sequence.


```python
import numpy as np
import torch
input_ids = torch.arange(4)
temperature = 1
np.set_printoptions(precision=2)

def one_hot(x):
    out = torch.nn.functional.one_hot(x)
    print(f"one_hot:\n{out.numpy()}\n")
    return out

def softmax(x):
    out = torch.nn.functional.softmax(x.to(torch.float32), dim=-1)
    print(f"sotmax:\n{out.numpy()}\n")
    return out

def randn_like(x):
    out = torch.randn_like(x.to(torch.float32))
    print(f"random noise:\n{out.numpy()}\n")
    return out
```


```python
noise_weight = random.uniform(0.25, 0.5) # variable amount of noise
clean_weight = 1 - noise_weight

hot = one_hot(input_ids)
w = clean_weight * hot + noise_weight * randn_like(hot)
soft_latents = softmax(w / temperature)

print(f"noise: {noise_weight:.3f} | input: {input_ids.tolist()} | corrupted: {soft_latents.argmax(-1).tolist()}")
```

    one_hot:
    [[1 0 0 0]
     [0 1 0 0]
     [0 0 1 0]
     [0 0 0 1]]
    
    random noise:
    [[-0.18 -0.95  0.94  0.8 ]
     [ 0.03  0.75 -0.32 -0.17]
     [-1.3   0.49  1.63 -1.23]
     [ 0.52  0.05  0.06  0.13]]
    
    sotmax:
    [[0.34 0.14 0.27 0.26]
     [0.19 0.47 0.17 0.18]
     [0.11 0.2  0.58 0.11]
     [0.23 0.19 0.2  0.38]]
    
    noise: 0.364 | input: [0, 1, 2, 3] | corrupted: [0, 1, 2, 3]


### "Soft" Tokens

To stabilize early training, we use a high temperature softmax to give a spread over the noisy tokens for a richer signal.

Temperature is annealead to 0 over a course of 500K steps, where this becomes equivalent to `argmax`.

```
Early in training (temperature = 1e-3)
softmax(noisy_one_hot_tokens / temperature) => [0.001, 0.92, 0.005, 0.074]

Late training (temperature = 1e-8)
softmax(noisy_one_hot_tokens / temperature) ~= argmax => [0, 1, 0, 0]
```

## Loss

By default, hard one-hot targets and CEL cause very high-variance gradiants, and it doesn't work very well.

A.k.a: For two rows of 15% and 99% noise, the 99% is a much harder and will **dominate the gradient.**


- Use a _Symmetric KL divergence_ between the target labels and logits.
    - Similar to CEL, but also penalises over-predicting wrong labels.  
- Use a scaling factor based on the amount of noise per row, so all rows have ~= loss

```python
p_prob = softmax(logits)    # predicted
q_prob = one_hot(input_ids) # ground-truth
sym_kl = symmetric_kl_divergence(p_prob, q_prob)

# As noise -> 0, view_scale -> 1, loss counts fully.
view_scale = (noise_weight / (vocab_size * clean_weight + noise_weight)
loss = (view_scale * sym_kl).sum()
```
