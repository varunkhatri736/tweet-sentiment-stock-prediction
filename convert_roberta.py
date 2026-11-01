"""Convert the RoBERTa-base encoder shipped in spaCy's en_core_web_trf 3.8.0
into a standard Hugging Face `RobertaModel` + tokenizer directory.

Why: huggingface.co is not reachable from the build environment, but the spaCy
model wheel (hosted on GitHub releases) contains full RoBERTa-base weights.
Note these weights were further fine-tuned by spaCy on OntoNotes (tagging,
parsing, NER), so they are *not* identical to vanilla `roberta-base`.

Usage: python convert_roberta.py <path/to/transformer/model> <out_dir>
"""
import io
import json
import sys
from pathlib import Path

import srsly
import torch
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

d = srsly.msgpack_loads(src.read_bytes())
shim = srsly.msgpack_loads(d["shims"][6][0])
sd = torch.load(io.BytesIO(shim["state"]), map_location="cpu", weights_only=False)
bpe = srsly.msgpack_loads(d["attrs"][3]["byte_bpe_processor"])

# --- tokenizer -------------------------------------------------------------
(out / "vocab.json").write_text(json.dumps(bpe["vocab"], ensure_ascii=False))
(out / "merges.txt").write_text(
    "#version: 0.2\n" + "\n".join(" ".join(m) for m in bpe["merges"]) + "\n"
)
tok = RobertaTokenizerFast(
    vocab=bpe["vocab"], merges=[tuple(m) for m in bpe["merges"]],
    model_max_length=512,
)
assert len(tok("Stocks rallied today")["input_ids"]) > 2, "tokenizer is broken"
tok.save_pretrained(out)

# --- weights ----------------------------------------------------------------
cfg = RobertaConfig(
    vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
    intermediate_size=3072, hidden_act="gelu", max_position_embeddings=514,
    type_vocab_size=1, layer_norm_eps=1e-5, pad_token_id=1,
)
# Name mapping is the exact inverse of curated-transformers 0.1.x
# `_convert_roberta_base_state` (hf_util.py), where the fused `mha.input`
# projection is torch.cat([query, key, value]).
P = "curated_encoder."
new = {}
e = P + "embeddings.inner."
new["embeddings.word_embeddings.weight"] = sd[e + "word_embeddings.weight"]
new["embeddings.position_embeddings.weight"] = sd[e + "position_embeddings.weight"]
new["embeddings.token_type_embeddings.weight"] = sd[e + "token_type_embeddings.weight"]
new["embeddings.LayerNorm.weight"] = sd[e + "layer_norm.weight"]
new["embeddings.LayerNorm.bias"] = sd[e + "layer_norm.bias"]
for i in range(12):
    s, t = f"{P}layers.{i}.", f"encoder.layer.{i}."
    qkv_w, qkv_b = sd[s + "mha.input.weight"], sd[s + "mha.input.bias"]
    for j, name in enumerate(["query", "key", "value"]):
        new[t + f"attention.self.{name}.weight"] = qkv_w[j * 768:(j + 1) * 768]
        new[t + f"attention.self.{name}.bias"] = qkv_b[j * 768:(j + 1) * 768]
    new[t + "attention.output.dense.weight"] = sd[s + "mha.output.weight"]
    new[t + "attention.output.dense.bias"] = sd[s + "mha.output.bias"]
    new[t + "attention.output.LayerNorm.weight"] = sd[s + "attn_output_layernorm.weight"]
    new[t + "attention.output.LayerNorm.bias"] = sd[s + "attn_output_layernorm.bias"]
    new[t + "intermediate.dense.weight"] = sd[s + "ffn.intermediate.weight"]
    new[t + "intermediate.dense.bias"] = sd[s + "ffn.intermediate.bias"]
    new[t + "output.dense.weight"] = sd[s + "ffn.output.weight"]
    new[t + "output.dense.bias"] = sd[s + "ffn.output.bias"]
    new[t + "output.LayerNorm.weight"] = sd[s + "ffn_output_layernorm.weight"]
    new[t + "output.LayerNorm.bias"] = sd[s + "ffn_output_layernorm.bias"]

model = RobertaModel(cfg, add_pooling_layer=False)
missing, unexpected = model.load_state_dict(new, strict=False)
assert not unexpected, unexpected
assert not missing, missing
model.save_pretrained(out)
print("saved to", out)
