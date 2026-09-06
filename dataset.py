import glob
import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = sorted(glob.glob(f"{feature_dir}/*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        caption = data["captions"][0] if isinstance(data["captions"], list) else data["captions"]
        return {
            "features": data["features"],
            "caption": caption,
            "all_captions": data["captions"]
        }


class DataCollator:
    def __init__(self, tokenizer, num_audio_tokens, max_length=512):
        self.tokenizer = tokenizer
        self.num_audio_tokens = num_audio_tokens
        self.max_length = max_length

    def __call__(self, batch):
        audio_features = torch.stack([item["features"] for item in batch])
        audio_placeholders = "<|audio_pad|>" * self.num_audio_tokens
        prompt_template = f"<|audio_start|>{audio_placeholders}<|audio_end|>\nTask: Describe the sound in this audio.\nAnswer: "
        prompts = [prompt_template for _ in batch]
        targets = [item["caption"] for item in batch]
        full_texts = [p + t + self.tokenizer.eos_token for p, t in zip(prompts, targets)]
        encodings = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        labels = input_ids.clone()
        for i, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            labels[i, :len(prompt_ids)] = -100
        labels[attention_mask == 0] = -100
        valid_labels = (labels != -100).sum()
        if valid_labels == 0:
            raise ValueError("batch contains zero target tokens! check max_length or prompt construction")
        return {
            "audio_features": audio_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }