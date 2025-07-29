# utils/trainer.py
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

class TrainerConfig:
    def __init__(self, max_iters, batch_size, learning_rate, lr_decay, warmup_iters,
                 lr_decay_iters, min_lr, betas, weight_decay, grad_clip,
                 device, dtype, compile, out_dir, log_interval,
                 eval_interval, eval_iters, eval_only, always_save_checkpoint, seed,
                 num_workers=4, debug=False, block_size=None):
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.device = device
        self.dtype = dtype
        self.compile = compile
        self.out_dir = out_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.eval_only = eval_only
        self.always_save_checkpoint = always_save_checkpoint
        self.seed = seed
        self.num_workers = num_workers
        self.debug = debug
        self.block_size = block_size  # sequence length expected by the model

class Trainer:
    def __init__(self, model, train_data, val_data, config):
        torch.manual_seed(config.seed)
        self.model = model.to(config.device)
        self.config = config

     
        if train_data.dim() == 1 and config.block_size:
            total_tokens = train_data.size(0)
            usable = (total_tokens // config.block_size) * config.block_size
            train_data = train_data[:usable].view(-1, config.block_size)
        if val_data.dim() == 1 and config.block_size:
            total_tokens = val_data.size(0)
            usable = (total_tokens // config.block_size) * config.block_size
            val_data = val_data[:usable].view(-1, config.block_size)

        # Prepare data loaders with proper shapes
        train_ds = TensorDataset(train_data)
        val_ds = TensorDataset(val_data)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=(config.device != 'cpu')
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(config.device != 'cpu')
        )
        self.train_iter = iter(self.train_loader)

        self.iter_num = 0
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate,
            betas=config.betas, weight_decay=config.weight_decay
        )
        if config.compile:
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

    def train_step(self):
        self.model.train()
        try:
            batch = next(self.train_iter)[0]
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)[0]

        # Batch is now shape (batch_size, block_size)
        x = batch.to(self.config.device)
        y = x
        logits, loss = self.model(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.iter_num += 1
        return loss.item(), self.optimizer.param_groups[0]['lr']

    def train(self):
        os.makedirs(self.config.out_dir, exist_ok=True)

        if self.config.debug:
            # Verbose header
            print("iter,loss,lr")
            for _ in range(self.config.max_iters):
                loss, lr = self.train_step()
                print(f"{self.iter_num},{loss:.6f},{lr:.3e}")
                if self.config.always_save_checkpoint and self.iter_num % self.config.eval_interval == 0:
                    ckpt = os.path.join(self.config.out_dir, f"ckpt_zoe_nano{self.iter_num}.pt")
                    torch.save(self.model.state_dict(), ckpt)
        else:
            from tqdm import tqdm
            pbar = tqdm(total=self.config.max_iters, desc="Training")
            while self.iter_num < self.config.max_iters:
                loss, lr = self.train_step()
                if self.iter_num % self.config.log_interval == 0:
                    pbar.set_postfix(loss=loss)
                if self.config.always_save_checkpoint and self.iter_num % self.config.eval_interval == 0:
                    ckpt = os.path.join(self.config.out_dir, f"ckpt_zoe_nano{self.iter_num}.pt")
                    torch.save(self.model.state_dict(), ckpt)
                pbar.update(1)
            pbar.close()
