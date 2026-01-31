import torch
import math
from model import GPT
from data import CodeDataset, get_batch
from config import Config

def estimate_loss(model, dataset, config):
    out = {}
    model.eval()
    for _ in range(config.eval_iters):
        X, Y = get_batch(dataset, config)
        with torch.no_grad():
            logits, loss = model(X, Y)
        out["loss"] = loss.item()
    model.train()
    return out["loss"]

def main():
    cfg = Config()
    device = cfg.device
    print(f"Running on {device}")

    # 1. Load Data
    dataset = CodeDataset(cfg)

    # 2. Initialize Model
    model = GPT(cfg).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # 4. Training Loop
    print("Starting training...")
    train_losses = []
    
    for iter in range(cfg.max_iters):
        
        # Evaluate loss
        if iter % cfg.eval_interval == 0:
            losses = estimate_loss(model, dataset, cfg)
            print(f"Step {iter}: Train Loss {losses:.4f}")

        # Sample a batch of data
        xb, yb = get_batch(dataset, cfg)

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 5. Save Model
    torch.save(model.state_dict(), "dev_gpt_model.pt")
    print("Training complete. Model saved to dev_gpt_model.pt")

if __name__ == "__main__":
    main()