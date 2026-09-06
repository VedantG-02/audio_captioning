import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import yaml
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from aac_datasets import Clotho
from transformers import WhisperProcessor, WhisperModel

def extract_and_save_audio_features(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("./features/train", exist_ok=True)
    os.makedirs("./features/eval", exist_ok=True)
                            
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-medium").encoder.to(device)
    whisper_model.eval()
    subsets = [("dev", "train"), ("eval", "eval")]
    for subset, local_split in subsets:
        ds = Clotho(root="./data", subset=subset, download=False)
        for idx in range(len(ds)):
            # print(idx)
            item = ds[idx]
            audio_tensor = item["audio"].squeeze(0)
            captions = item["captions"]

            orig_sr = item.get("sr", 44100)
            if orig_sr != 16000:
                resampler = T.Resample(orig_freq=orig_sr, new_freq=16000)
                audio_tensor = resampler(audio_tensor)

            inputs = whisper_processor(
                audio_tensor.numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(device)
            with torch.no_grad():
                features = whisper_model(inputs).last_hidden_state     # [1, 1500, 768]
                features = features.transpose(1, 2)
                to_pool = config["to_pool"]
                if to_pool:                                            # [1, 300, 768]  
                    kernel_size = 5
                    stride = 5
                else:                                                  # [1, 1500, 768]
                    kernel_size = 1
                    stride = 1
                pooled_features = F.avg_pool1d(features, kernel_size=kernel_size, stride=stride)
                pooled_features = pooled_features.transpose(1, 2).squeeze(0).cpu()
            file_path = f"./features/{local_split}/{idx}.pt"
            torch.save({
                "features": pooled_features.half(),
                "captions": captions
            }, file_path)

            if (idx + 1) % 100 == 0:
                print(f"[{local_split.upper()}] processed {idx + 1}/{len(ds)} items")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

    extract_and_save_audio_features(config)