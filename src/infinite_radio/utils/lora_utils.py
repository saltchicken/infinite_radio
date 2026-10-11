import torch

def _mods(prefix):
    return [(f"{prefix}self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + \
           [(f"{prefix}mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]


def load_ckpt(path, map_location="cuda"):
    from safetensors.torch import load_file
    from safetensors import safe_open

    t = load_file(path, device=str(map_location))
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}

    prefix = "nar_" if any(".nar_self_attn." in k for k in t) else ""
    layers = sorted({int(k.split(".")[1]) for k in t if k.startswith("layers.")})
    lora = []

    for L in layers:
        for blk, proj in _mods(prefix):
            lora += [
                t[f"layers.{L}.{blk}.{proj}.lora_A"],
                t[f"layers.{L}.{blk}.{proj}.lora_B"]
            ]

    out = {"lora": lora, "rank": int(meta.get("rank", lora[0].shape[0]))}
    if prefix:
        out["io"] = {
            m: {
                k.split(".", 1)[1]: v
                for k, v in t.items()
                if k.startswith(m + ".")
            } for m in ("vae2llm", "llm2vae")
        }
    return out


@torch.no_grad()
def merge_lora(bb, attn_name, mlp_name, tensors, scale=1.0, dev="cuda"):
    it = iter(tensors)
    n_merged = 0
    for layer in bb.layers:
        for mod, names in ((getattr(layer, attn_name), ("q_proj", "k_proj", "v_proj", "o_proj")),
                           (getattr(layer, mlp_name), ("gate_proj", "up_proj", "down_proj"))):
            for n in names:
                A = next(it).to(dev).float()
                B = next(it).to(dev).float()
                lin = getattr(mod, n)
                lin.weight.add_((scale * (B @ A)).to(lin.weight.dtype))
                n_merged += 1
    return n_merged
