import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

import yaml
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from model import AudioLLM
from dataset import CaptionDataset, DataCollator


def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    targets = config['lora']['targets']
    suffix = ""
    for target in ["q_proj", "v_proj", "k_proj", "o_proj"]:
        if target in targets:
            suffix += target[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = f"./checkpoint_bridge_{config['bridge']}_r_{config['lora']['r']}_a_{config['lora']['alpha']}_t_{suffix}"
    
    print(f"loading model with LoRA | r={config['lora']['r']} | alpha={config['lora']['alpha']}\n")
    model = AudioLLM(config).to(device)
    train_dataset = CaptionDataset(f"{config['feature_dir']}/train")
    if config['to_pool']:
        raw_feature_len  = 300
    else:
        raw_feature_len  = 1500
    num_audio_tokens = model.get_num_audio_tokens(raw_feature_len) 
    collator = DataCollator(model.tokenizer, num_audio_tokens=num_audio_tokens, max_length=config['max_length'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    model.train()
    print("Training starts ...\n")
    for epoch in range(config['epochs']):
        print(f"> Epoch {epoch+1:2d} starts\n")
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            features = batch["audio_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(features, input_ids, attention_mask, labels)
                loss = outputs.loss
            # sanity check
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"NaN Loss detected at Epoch {epoch+1}, Step {step+1}!")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 40 == 0:
                print(f"    Epoch [{epoch+1:2d}/{config['epochs']}] | Step [{step+1:3d}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_loss = total_loss/len(train_loader)
        print(f"\n> Epoch {epoch+1:2d} | Average Loss: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()