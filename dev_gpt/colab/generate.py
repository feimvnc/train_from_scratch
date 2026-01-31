import torch
from model import GPT
from config import Config
import tiktoken

def generate_code(prompt):
    cfg = Config()
    device = cfg.device
    
    # Load Model
    model = GPT(cfg)
    model.load_state_dict(torch.load("dev_gpt_model.pt"))
    model.to(device)
    model.eval()
    
    # Tokenizer
    enc = tiktoken.get_encoding(cfg.tokenizer_name)
    
    # Encode prompt
    idx = enc.encode(prompt, allowed_special={'<|endoftext|>'})
    idx = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        for _ in range(100): # Generate 100 tokens
            logits, _ = model(idx)
            logits = logits[:, -1, :] # focus on last step
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == enc.eot_token:
                break
                
    print(enc.decode(idx[0].tolist()))

if __name__ == "__main__":
    generate_code("def calculate_sum(a, b):\n    ")