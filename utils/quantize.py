import torch
import os
from model import GPTConfig, GPT
from trainer import quantize_ternary


CKPT_IN     = "model/ckpt_zoe_nano10000.pt"
CKPT_OUT    = "model/ckpt_zoe_nano10000_quant.pt"
VOCAB_SIZE  = 99998
BLOCK_SIZE  = 512
N_LAYER     = 2
N_HEAD      = 4
N_EMBD      = 256

# --- Carrega checkpoint original ---
print(f"🔄 Carregando checkpoint {CKPT_IN}")
state_dict = torch.load(CKPT_IN, map_location="cpu")

# --- Reconstrói arquitetura ---
conf = GPTConfig(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_embd=N_EMBD
)
model = GPT(conf)
model.load_state_dict(state_dict)
model.eval()

# --- Aplica quantização ternária ---
print("⚙️  Aplicando quantização ternária aos pesos...")
quantize_ternary(model)

# --- Salva novo checkpoint quantizado ---
os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)
torch.save(model.state_dict(), CKPT_OUT)
print(f"✅ Checkpoint quantizado salvo em {CKPT_OUT}")

