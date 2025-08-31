import argparse
import time
import math
import warnings

warnings.filterwarnings('ignore')
import os
import sys
import torch

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from contextlib import nullcontext
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from model.model_vlm import MiniMindVLM, VLMConfig
from dataset.lm_dataset import VLMDataset

def Logger(content):
    print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask, pixel_values) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X, pixel_values=pixel_values)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if wandb is not None:
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0:
            model.eval()
            moe_path = '_moe' if model_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_vlm_{model_config.hidden_size}{moe_path}.pth'
            state_dict = model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}  # Save in half precision
            torch.save(clean_state_dict, ckp)
            model.train()


def init_model(model_config: VLMConfig):
    tokenizer = AutoTokenizer.from_pretrained('../model', use_fast=True)
    moe_path = '_moe' if model_config.use_moe else ''
    # Load pure language model weights
    ckp = f'{args.save_dir}/llm_{model_config.hidden_size}{moe_path}.pth'
    model = MiniMindVLM(model_config, vision_model_path="../model/vision_model/clip-vit-base-patch16")
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    # Freeze all parameters except vision_proj
    for name, param in model.named_parameters():
        if 'vision_proj' not in name:
            param.requires_grad = False

    Logger(f'VLM trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    _, preprocess = model.vision_encoder, model.processor
    return model.to(args.device), tokenizer, preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Pretrain")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_data.jsonl")
    parser.add_argument("--images_path", type=str, default="../dataset/pretrain_images")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=640, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    args = parser.parse_args()

    model_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                             max_seq_len=args.max_seq_len)
    max_seq_len = model_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-V Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer, preprocess = init_model(model_config)

    train_ds = VLMDataset(args.data_path, args.images_path, tokenizer, preprocess=preprocess,
                          image_special_token=model_config.image_special_token,
                          max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
