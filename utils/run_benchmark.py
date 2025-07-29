import time
import json
import torch
from tqdm import tqdm
from model import GPTConfig, GPT
from tokenizer import encode, decode, sp

CHECKPOINT_PATH = 'model/ckpt_zoe_nano10000.pt'
BLOCK_SIZE      = 512
MAX_NEW_TOKENS  = 150

def load_model():
    state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
    config = GPTConfig(
        vocab_size=99998,
        block_size=BLOCK_SIZE,
        n_layer=2,
        n_head=4,
        n_embd=256
    )
    model = GPT(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def bench_tokenizer(lines, n_iters=1000):
    print("🔍 Benchmarking tokenizer...")
    start = time.perf_counter()
    total_tokens = 0
    for i, line in enumerate(lines):
        if i >= n_iters: break
        ids = encode(line)
        total_tokens += len(ids)
    elapsed = time.perf_counter() - start
    print(f"Tokenizer: {total_tokens} tokens in {elapsed:.2f}s → {total_tokens/elapsed:.1f} tok/s")

def bench_generation(model, prompts):
    print("\n🚀 Benchmarking generation:")
    for prompt in prompts:
        ids = encode(prompt)
        x = torch.tensor([ids], dtype=torch.long)
        # warmup
        with torch.no_grad():
            _ = model.generate(x, max_new_tokens=5)
        # timed
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=MAX_NEW_TOKENS)
        dt = time.perf_counter() - t0
        gen_tokens = len(out[0]) - len(ids)
        print(f"- Prompt len {len(ids):3d}, generated {gen_tokens:3d} tok in {dt:.2f}s → {gen_tokens/dt:.1f} tok/s")

def diag_forward(model):
    print("\n📈 Diagnóstico de forward pass:")
    dummy = torch.randint(0, len(sp), (1, BLOCK_SIZE))
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, loss = model(dummy, dummy)
    dt = time.perf_counter() - t0
    print(f"Forward+loss ({BLOCK_SIZE} tokens): {dt:.3f}s")

if __name__ == "__main__":
    sample_lines = []
    with open('datasets/dataset_tokenized.jsonl', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000: break
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and 'input_ids' in obj:
                    txt = decode(obj['input_ids'])
                else:
                    txt = line.strip()
            except:
                txt = line.strip()
            sample_lines.append(txt)

    model = load_model()
    bench_tokenizer(sample_lines, n_iters=500)
    bench_generation(model, [
        "Olá, como estás?",
        "Explique de forma simples o que é um transformador em NLP.",
        "Diz-me três factos sobre a física quântica.",
    ])
    diag_forward(model)


