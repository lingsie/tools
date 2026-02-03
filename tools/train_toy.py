#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed




class TinyModel(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )


    def forward(self, x):
        return self.net(x)




def get_args():
    parser = argparse.ArgumentParser()


    # DeepSpeed launcher 必传的 local_rank
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank for distributed training on gpus"
    )


    # 自己的参数
    parser.add_argument("--train-steps", type=int, default=50)


    # 为了不报 unknown，把后端传进来的这些都声明一下
    parser.add_argument("--zero-stage", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)


    # 让 DeepSpeed 注入它自己的参数（--deepspeed, --deepspeed_config 等）
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args




def main():
    args = get_args()
    torch.manual_seed(42)


    hidden = 256
    model = TinyModel(hidden=hidden)
    criterion = nn.MSELoss()


    # 建一个普通 AdamW 优化器交给 DeepSpeed 包
    base_optimizer = optim.AdamW(model.parameters(), lr=args.lr)


    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        optimizer=base_optimizer,
    )


    # 关键：用和模型同样的 dtype & device 生成数据，避免 Float vs BFloat16 冲突
    device = model_engine.device
    param_dtype = next(model_engine.parameters()).dtype


    for step in range(1, args.train_steps + 1):
        batch_size = 8


        x = torch.randn(
            batch_size,
            hidden,
            device=device,
            dtype=param_dtype,
        )
        target = torch.zeros(
            batch_size,
            hidden,
            device=device,
            dtype=param_dtype,
        )


        y = model_engine(x)
        loss = criterion(y, target)


        model_engine.backward(loss)
        model_engine.step()


        if model_engine.global_rank == 0 and (step % 5 == 0 or step == 1):
            # 这里的格式专门兼容你后端的 regex：step=xxx loss=xxx
            print(f"step={step} loss={loss.item():.6f}", flush=True)




if __name__ == "__main__":
    main()



