# utils/chat.py

import torch
from model import GPTConfig, GPT
from tokenizer import sp, encode, decode
import readline

CKPT_PATH      = 'model/ckpt_zoe_nano10000.pt'
BLOCK_SIZE     = 512
MAX_CONTEXT    = 1200
MAX_NEW_TOKENS = 150

sd = torch.load(CKPT_PATH, map_location='cpu')
conf = GPTConfig(
    vocab_size=99998,
    block_size=BLOCK_SIZE,
    n_layer=2,
    n_head=4,
    n_embd=256
)
model = GPT(conf)
model.load_state_dict(sd)
model.eval()

history_tokens = []

def build_context(tokens_list):
    flat = []
    for seq in tokens_list:
        flat.extend(seq)
    return flat[-MAX_CONTEXT:]

print("ZoeNano v0.0.1\n")
while True:
    user_text = input("You: ").strip()
    if user_text.lower() in ('sair', 'exit', 'quit'):
        print("Até à próxima!")
        break

    utoks = encode(user_text)
    history_tokens.append(utoks)

    ctx = build_context(history_tokens)
    x = torch.tensor([ctx], dtype=torch.long)

    gen = x.clone()
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits, _ = model(gen)
            last = logits[:, -1, :]
            nxt = torch.argmax(last, dim=-1, keepdim=True)
            gen = torch.cat((gen, nxt), dim=1)

    full = gen[0].tolist()
    new = full[len(ctx):]
    reply = decode(new).strip()

    if not reply:
        reply = "[sem resposta]"

    print("Zoe:", reply, "\n")

    history_tokens.append(encode(reply))

