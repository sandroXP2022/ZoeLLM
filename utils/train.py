# utils/train.py
import os
import json
import torch
from model import GPTConfig, GPT
from trainer import Trainer, TrainerConfig

# ---------------------- VARIÁVEL DE DEBUG ----------------------
DEBUG = 0  # 1 = log linha a linha, 0 = tqdm (barra de progresso)

# ---------------------- NÃO TESTADO ----------------------
# def quantize_ternary(model):
#    with torch.no_grad():
#        for p in model.parameters():
#            if p.requires_grad:
#                p.data.copy_(torch.sign(p.data))


def main():
    # ---------------------- CONFIGS GERAIS ----------------------
    out_dir = 'model/'
    eval_interval = 10000
    log_interval = 10
    eval_iters = 500
    eval_only = False
    always_save_checkpoint = True
    init_from = 'scratch'

    # ---------------------- DADOS ----------------------
    dataset = 'datasets/dataset_tokenized.jsonl'
    batch_size = 4
    block_size = 512

    # ---------------------- ARQUITETURA ----------------------
    n_layer, n_head, n_embd = 2, 4, 256
    dropout = 0.0

    # ---------------------- OTIMIZAÇÃO ----------------------
    learning_rate = 1e-2
    max_iters = 10000
    weight_decay = 4e-2
    beta1, beta2 = 0.9, 0.95
    grad_clip = 1.0
    decay_lr = True
    warmup_iters = 50
    lr_decay_iters = 5000
    min_lr = 4e-2

    # ---------------------- EXECUÇÃO ----------------------
    device = 'cpu'
    dtype = 'bfloat16'
    compile_model = False
    seed = 5555

    # ---------------------- CARREGAR DADOS ----------------------
    torch.manual_seed(seed)
    print("📚 Carregando dataset JSONL tokenizado...")
    all_tokens = []
    with open(dataset, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line.strip())
                tokens = obj.get('input_ids', [])
                if isinstance(tokens, list):
                    all_tokens.extend(tok for tok in tokens if isinstance(tok, int))
            except json.JSONDecodeError:
                print(f"Ignorando linha {line_num}: JSON inválido")

    data = torch.tensor(all_tokens, dtype=torch.long)
    n = data.size(0)
    print(f"✅ Tokens carregados: {n}")

    vocab_size = int(torch.max(data).item()) + 1 if n > 0 else 0
    print(f"📈 Vocab size ajustado para {vocab_size}")

    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]
    
    train_data_tensor = train_data.clone().detach().long()
    val_data_tensor = val_data.clone().detach().long()

    
    train_data_tensor = train_data_tensor[:train_data_tensor.size(0) // block_size * block_size]
    val_data_tensor = val_data_tensor[:val_data_tensor.size(0) // block_size * block_size]

    train_data_tensor = train_data_tensor.view(-1, block_size)
    val_data_tensor = val_data_tensor.view(-1, block_size)

    print(f"🔧 Tokens treino={train_data.size(0)} | val={val_data.size(0)}")

    # ---------------------- CONSTRUIR MODELO ----------------------
    gptconf = GPTConfig(vocab_size, block_size, n_layer, n_head, n_embd)
    model = GPT(gptconf)

    # ---------------------- CONFIGURAR TREINADOR ----------------------
    tconf = TrainerConfig(
        max_iters, batch_size, learning_rate,
        decay_lr, warmup_iters, lr_decay_iters, min_lr,
        (beta1, beta2), weight_decay, grad_clip,
        device, dtype, compile_model, out_dir,
        log_interval, eval_interval, eval_iters,
        eval_only, always_save_checkpoint, seed,
        debug=DEBUG,
        num_workers=8, 
        block_size=block_size
    )
    trainer = Trainer(model, train_data, val_data, tconf)

    # ---------------------- INICIAR TREINO ----------------------
    print("🚀 TREINAMENTO INICIADO")
    trainer.train()


if __name__ == "__main__":
    main()

