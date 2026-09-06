import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import yaml
import torch
import evaluate
# from aac_metrics import Evaluate as AACEvaluate
from model import AudioLLM
from dataset import CaptionDataset


def eval():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    targets = config['lora']['targets']
    suffix = ""
    for target in ["q_proj", "v_proj", "k_proj", "o_proj"]:
        if target in targets:
            suffix += target[0]

    checkpoint_path = f"./checkpoint_bridge_{config['bridge']}_r_{config['lora']['r']}_a_{config['lora']['alpha']}_t_{suffix}"
    print(f"loading ckpt from {checkpoint_path}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AudioLLM(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    eval_dataset = CaptionDataset(f"{config['feature_dir']}/eval")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    # spider = AACEvaluate(metrics=["spider"])

    predictions = []
    references = []

    if config["to_pool"]:
        raw_feature_len  = 300
    else:
        raw_feature_len  = 1500
    num_audio_tokens = model.get_num_audio_tokens(raw_feature_len)
    audio_placeholders = "<|audio_pad|>" * num_audio_tokens
    prompt = f"<|audio_start|>{audio_placeholders}<|audio_end|>\nTask: Describe the sound in this audio.\nAnswer: "
    prompt_encodings = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_encodings.input_ids.to(device)

    start_indices = (prompt_ids[0] == model.audio_start_id).nonzero(as_tuple=True)[0]
    end_indices = (prompt_ids[0] == model.audio_end_id).nonzero(as_tuple=True)[0]
    
    if len(start_indices) == 0 or len(end_indices) == 0:
        raise ValueError("Audio special tokens not found in tokenized evaluation prompt.")

    s_idx = start_indices[0].item() + 1
    e_idx = end_indices[0].item()

    print("Running Inference ...\n")
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            item = eval_dataset[i]
            features = item["features"].unsqueeze(0).to(device)
            projected_audio = model.bridge(features.to(torch.float32)).to(torch.bfloat16)
            text_embeds = model.llm.get_input_embeddings()(prompt_ids).to(torch.bfloat16)
            num_placeholders = e_idx - s_idx
            inputs_embeds = torch.cat([
                text_embeds[:, :s_idx],
                projected_audio[:, :num_placeholders],
                text_embeds[:, e_idx:]
            ], dim=1)
            outputs = model.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
            pred_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred_text.strip())
            references.append(item["all_captions"])

            if (i + 1) % 100 == 0:
                print(f"    Evaluated {i+1:4d}/{len(eval_dataset)} samples")

    bleu_results = bleu.compute(predictions=predictions, references=references)
    rouge_results = rouge.compute(predictions=predictions, references=[r[0] for r in references])
    meteor_results = meteor.compute(predictions=predictions, references=references)

    # predictions = [pred.replace("\n", " ").strip() for pred in predictions]
    # spider_results, _ = spider(candidates=predictions, mult_references=references)

    print("\n======== FINAL EVALUATION RESULTS ========")
    print(f"BLEU-4 Score : {bleu_results['bleu'] * 100:.2f}")
    print(f"ROUGE-L Score: {rouge_results['rougeL'] * 100:.2f}")
    print(f"METEOR Score : {meteor_results['meteor'] * 100:.2f}")
    # print(f"SPIDEr Score : {float(spider_results['spider']) * 100:.2f}")
    print("===========================================")

if __name__ == "__main__":
    eval()