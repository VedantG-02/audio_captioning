import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Blip2QFormerModel
from peft import get_peft_model, LoraConfig, TaskType


class MLPBridge(nn.Module):
    def __init__(self, llm_dim, whisper_dim=768, hidden_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(whisper_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, llm_dim)
        )

    def forward(self, x):
        return self.proj(x)


class QFormerBlock(nn.Module):
    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, query, audio_kv):
        q_norm = self.norm1(query)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        query = query + sa_out
        q_norm = self.norm2(query)
        ca_out, _ = self.cross_attn(query=q_norm, key=audio_kv, value=audio_kv)
        query = query + ca_out
        q_norm = self.norm3(query)
        ffn_out = self.ffn(q_norm)
        query = query + ffn_out
        return query


class WindowQFormerBridge(nn.Module):
    def __init__(self, pretrained_model, llm_dim, whisper_dim=768, use_pretrained=False, window_size=50, queries_per_window=10):
        super().__init__()
        self.use_pretrained = use_pretrained
        self.window_size = window_size
        self.queries_per_window = queries_per_window
        if self.use_pretrained:
            print(f"\n> initializing pretrained BLIP-2 qformer from: {pretrained_model}")
            self.qformer = Blip2QFormerModel.from_pretrained(pretrained_model)
            qformer_dim = self.qformer.config.hidden_size  # 768
        else:
            print("\n> initializing window level scratch qformer")
            qformer_dim = 768
            self.blocks = nn.ModuleList([
                QFormerBlock(d_model=qformer_dim, nhead=8, dim_feedforward=qformer_dim * 4, dropout=0.1)
                for _ in range(2)
            ])
            self.norm_out = nn.LayerNorm(qformer_dim)
        self.audio_proj = nn.Linear(whisper_dim, qformer_dim) if whisper_dim != qformer_dim else nn.Identity()
        self.query_tokens = nn.Parameter(torch.randn(1, queries_per_window, qformer_dim) * 0.02)
        self.proj = nn.Linear(qformer_dim, llm_dim)

    def forward(self, audio_features):
        B, T, _ = audio_features.shape
        audio_kv = self.audio_proj(audio_features)  # [B, T, qformer_dim]
        pad_len = (self.window_size - (T % self.window_size)) % self.window_size
        if pad_len > 0:
            audio_kv = F.pad(audio_kv, (0, 0, 0, pad_len))
            T_padded = T + pad_len
        else:
            T_padded = T
        num_windows = T_padded // self.window_size
        audio_windows = audio_kv.view(B * num_windows, self.window_size, -1)
        queries = self.query_tokens.expand(B * num_windows, -1, -1)
        if self.use_pretrained:
            qformer_outputs = self.qformer(
                query_embeds=queries,
                encoder_hidden_states=audio_windows
            )
            queries_out = qformer_outputs.last_hidden_state
        else:
            for block in self.blocks:
                queries = block(queries, audio_windows)
            queries_out = self.norm_out(queries)
        _, K, D = queries_out.shape
        queries_out = queries_out.view(B, num_windows * K, D)
        target_seq_len = min(T, num_windows * K)
        queries_out = queries_out[:, :target_seq_len, :]
        projected_audio = self.proj(queries_out)  # [B, target_seq_len, llm_dim]
        return projected_audio


class AudioLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_id = "Qwen/Qwen2.5-1.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        special_tokens = {
            "additional_special_tokens": ["<|audio_start|>", "<|audio_end|>", "<|audio_pad|>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.audio_start_id = self.tokenizer.convert_tokens_to_ids("<|audio_start|>")
        self.audio_end_id = self.tokenizer.convert_tokens_to_ids("<|audio_end|>")
        self.audio_pad_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        base_llm = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16
        )
        base_llm.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad(): 
            embed_weights = base_llm.get_input_embeddings().weight
            num_new = len(special_tokens["additional_special_tokens"])
            mean_embed = embed_weights[:-num_new].mean(dim=0)
            for token in special_tokens["additional_special_tokens"]:
                tok_id = self.tokenizer.convert_tokens_to_ids(token)
                embed_weights[tok_id] = mean_embed.clone()
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            lora_dropout=config['lora']['dropout'],
            target_modules=config['lora']['targets']
        )
        self.llm = get_peft_model(base_llm, peft_config)
        llm_dim = base_llm.config.hidden_size
        bridge_type = config['bridge']
        whisper_dim = config['whisper_dim']
        if bridge_type == "mlp":
            print("\n> using mlp bridge")
            hidden_dim = config['hidden_dim']
            self.bridge = MLPBridge(
                llm_dim=llm_dim, 
                whisper_dim=whisper_dim, 
                hidden_dim=hidden_dim
            )
        elif bridge_type == "qformer":
            print("\n> using qformer bridge")
            qformer_cfg = config['qformer']
            self.bridge = WindowQFormerBridge(
                llm_dim=llm_dim, 
                whisper_dim=whisper_dim,
                use_pretrained=qformer_cfg['use_pretrained'],
                pretrained_model=qformer_cfg['pretrained_model'],
                window_size=qformer_cfg['window_size'],
                queries_per_window=qformer_cfg['queries_per_window']
            )
        
    def forward(self, audio_features, input_ids, attention_mask, labels):
        if torch.isnan(audio_features).any() or torch.isinf(audio_features).any():
            raise ValueError("SANITY CHECK FAILED: audio_features contain NaN/Inf.")
        audio_features = audio_features.to(torch.float32)
        projected_audio = self.bridge(audio_features).to(torch.bfloat16)
        text_embeds = self.llm.get_input_embeddings()(input_ids).to(torch.bfloat16)
        inputs_embeds_list = []
        for batch_idx in range(input_ids.shape[0]):
            ids = input_ids[batch_idx]
            s_indices = (ids == self.audio_start_id).nonzero(as_tuple=True)[0]
            e_indices = (ids == self.audio_end_id).nonzero(as_tuple=True)[0]
            if len(s_indices) == 0 or len(e_indices) == 0:
                raise ValueError("Sequence truncated! <|audio_start|> or <|audio_end|> missing. Increase max_length in config.yaml.")
            s_idx = s_indices[0].item() + 1
            e_idx = e_indices[0].item()
            num_placeholders = e_idx - s_idx
            sample_embeds = torch.cat([
                text_embeds[batch_idx, :s_idx],
                projected_audio[batch_idx, :num_placeholders],
                text_embeds[batch_idx, e_idx:]
            ], dim=0)
            inputs_embeds_list.append(sample_embeds)
        inputs_embeds = torch.stack(inputs_embeds_list, dim=0)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs