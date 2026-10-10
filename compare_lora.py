import argparse
import json
import torch
from pathlib import Path
from yue2 import YuE2Pipeline

# --- LoRA Loading Helpers ---
def _mods(prefix): 
    return [(f"{prefix}self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + \
           [(f"{prefix}mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]

def load_ckpt(path, map_location="cuda"):
    from safetensors.torch import load_file
    from safetensors import safe_open
    
    t = load_file(path, device=str(map_location))
    with safe_open(path, "pt") as f: 
        meta = f.metadata() or {}
        
    prefix = ""
    layers = sorted({int(k.split(".")[1]) for k in t if k.startswith("layers.")})
    lora = []
    
    for L in layers:
        for blk, proj in _mods(prefix): 
            lora += [t[f"layers.{L}.{blk}.{proj}.lora_A"], t[f"layers.{L}.{blk}.{proj}.lora_B"]]
            
    return {"lora": lora, "rank": int(meta.get("rank", lora[0].shape[0]))}

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
                # Fold LoRA deltas (W += scale * B @ A) into the base weights
                lin.weight.add_((scale * (B @ A)).to(lin.weight.dtype))
                n_merged += 1
    return n_merged
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="A/B Test a YuE2 song with and without a LoRA.")
    parser.add_argument("--config", type=Path, default="songs/acid_techno.json", help="Path to a song JSON config (e.g., songs/acid_techno.json)")
    parser.add_argument("--ar-lora", type=Path, required=True, help="Path to the AR LoRA safetensors file")
    parser.add_argument("--ar-scale", type=float, default=1.0, help="Scale (strength) of the LoRA")
    parser.add_argument("--output", type=Path, default="outputs/comparison")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for both generations")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Parse the JSON config deterministically
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Flatten lyrics (always pick the first layout for a true 1:1 comparison)
    if isinstance(config.get("lyrics"), list):
        chosen_lyrics = config["lyrics"][0] 
        if isinstance(chosen_lyrics, list):
            config["lyrics"] = "\n".join(chosen_lyrics)
        else:
            config["lyrics"] = chosen_lyrics

    # Flatten style variables (always pick the first option)
    if "choices" in config:
        choices = config.pop("choices")
        if isinstance(config.get("style"), str):
            for key, options in choices.items():
                config["style"] = config["style"].replace(f"{{{key}}}", str(options[0]))

    # Set parameters
    config["seed"] = args.seed
    config.pop("id", None) 

    print(f"\n======================================")
    print(f"🎵 Comparison Test Parameters")
    print(f"======================================")
    print(f"Style: {config['style']}")
    print(f"Lyrics:\n{config['lyrics']}")
    print(f"Seed: {config['seed']}")
    print(f"COT Mode: {config.get('cot', 'off')}")
    print(f"AR LoRA Scale: {args.ar_scale}")
    print(f"======================================\n")

    print("Loading YuE2 Base Model...")
    with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="cuda") as pipe:
        
        # --- PHASE 1: Base Generation ---
        print("\n>>> Generating PHASE 1: Base Model (No LoRA)...")
        base_song = pipe(**config)
        base_out = args.output / "01_base_model"
        base_out.mkdir(exist_ok=True)
        base_song.save_artifacts(base_out)
        print(f"✅ Base track saved to: {base_out}/audio.flac")

        # --- PHASE 2: Apply LoRA ---
        print(f"\n>>> Injecting AR LoRA from {args.ar_lora} into the active model (scale: {args.ar_scale})...")
        lora_data = load_ckpt(str(args.ar_lora), "cuda")
        base_model = pipe._load_model().model
        n_merged = merge_lora(base_model, "self_attn", "mlp", lora_data["lora"], scale=args.ar_scale, dev="cuda")
        print(f"✅ Merged {n_merged} AR LoRA linear layers successfully.")

        # --- PHASE 3: LoRA Generation ---
        print("\n>>> Generating PHASE 2: Model + LoRA (Same seed and params)...")
        lora_song = pipe(**config)
        lora_out = args.output / "02_lora_model"
        lora_out.mkdir(exist_ok=True)
        lora_song.save_artifacts(lora_out)
        print(f"✅ LoRA track saved to: {lora_out}/audio.flac")

    print("\n🎉 Comparison complete! You can now listen to both versions in the outputs/comparison folder.")

if __name__ == "__main__":
    main()
