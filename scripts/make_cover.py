import argparse 
import yaml
import torch 
import json
import random
from pathlib import Path 
from datetime import datetime
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
        
    prefix = "nar_" if any(".nar_self_attn." in k for k in t) else "" 
    layers = sorted({int(k.split(".")[1]) for k in t if k.startswith("layers.")}) 
    lora = [] 
    
    for L in layers: 
        for blk, proj in _mods(prefix): 
            lora += [t[f"layers.{L}.{blk}.{proj}.lora_A"], t[f"layers.{L}.{blk}.{proj}.lora_B"]] 
            
    out = {"lora": lora, "rank": int(meta.get("rank", lora[0].shape[0]))} 
    if prefix: 
        out["io"] = {m: {k.split(".", 1)[1]: v for k, v in t.items() if k.startswith(m + ".")} for m in ("vae2llm", "llm2vae")} 
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
                # Fold LoRA deltas (W += scale * B @ A) into the base weights 
                lin.weight.add_((scale * (B @ A)).to(lin.weight.dtype)) 
                n_merged += 1 
    return n_merged 
# ---------------------------- 

def main(): 
    parser = argparse.ArgumentParser(description="Generate a song cover using a config file.") 
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config.yaml file") 
    cmd_args = parser.parse_args() 

    with open(cmd_args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    abc_path = Path(config.get("abc"))
    style = config.get("style")
    lyrics_input = config.get("lyrics")
    output_base = Path(config.get("output", "outputs"))
    
    ar_lora_path = Path(config.get("ar_lora")) if config.get("ar_lora") else None
    ar_scale = float(config.get("ar_scale", 1.0))
    nar_lora_path = Path(config.get("nar_lora")) if config.get("nar_lora") else None
    nar_scale = float(config.get("nar_scale", 1.0))
    
    # --- Random Seed Logic ---
    raw_seed = config.get("seed", -1)
    # Check if the seed is explicitly random, null, or -1
    if raw_seed in [-1, None, "random"]:
        seed = random.randint(0, 4294967295) # Max 32-bit unsigned integer
    else:
        seed = int(raw_seed)
    
    cot = config.get("cot", "full")

    # 1. Load ABC content 
    if not abc_path.exists(): 
        print(f"❌ Error: ABC file not found at {abc_path}") 
        return 
    abc_content = abc_path.read_text(encoding="utf-8") 

    # --- Create output folder with timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"{abc_path.stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save parameters to the new output folder (will save the exact randomized seed used)
    saved_params = {
        "config_file_used": cmd_args.config,
        "abc_path": str(abc_path),
        "style": style,
        "lyrics_input": lyrics_input,
        "ar_lora": str(ar_lora_path) if ar_lora_path else None,
        "ar_scale": ar_scale,
        "nar_lora": str(nar_lora_path) if nar_lora_path else None,
        "nar_scale": nar_scale,
        "seed": seed, # This will log the actual random seed used so you can reproduce it later!
        "cot": cot,
        "timestamp": timestamp
    }
    
    with open(output_dir / "parameters.yaml", "w", encoding="utf-8") as f:
        yaml.dump(saved_params, f, default_flow_style=False, sort_keys=False)

    # 2. Load Lyrics 
    lyrics_path = Path(lyrics_input) 
    if lyrics_path.is_file(): 
        lyrics_content = lyrics_path.read_text(encoding="utf-8") 
    elif lyrics_input.endswith(".txt"): 
        print(f"❌ Error: Lyrics file '{lyrics_input}' not found. Check your spelling.") 
        return 
    else: 
        lyrics_content = lyrics_input 

    # 3. Print out all parameters being used 
    print(f"\n======================================") 
    print(f"🎸 Generating Cover (Config: {cmd_args.config})") 
    print(f"======================================") 
    print(f"ABC Source : {abc_path}") 
    print(f"Output Dir : {output_dir}") 
    print(f"Seed       : {seed}") 
    print(f"CoT Mode   : {cot}") 
    
    if ar_lora_path: 
        print(f"AR LoRA    : {ar_lora_path} (Scale: {ar_scale})") 
    if nar_lora_path: 
        print(f"NAR LoRA   : {nar_lora_path} (Scale: {nar_scale})") 
        
    print(f"\n--- Style ---") 
    print(style) 
    
    print(f"\n--- Lyrics ---") 
    print(lyrics_content.strip()) 
    print(f"======================================\n") 

    print("Loading YuE2 Base Model...") 
    with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="cuda") as pipe: 
        
        # --- Inject LoRAs if specified --- 
        if ar_lora_path or nar_lora_path: 
            print("\n>>> Injecting LoRAs into the model...") 
            top_model = pipe._load_model() 
            bb = top_model.model 
            
            if ar_lora_path and ar_lora_path.exists(): 
                print(f"Loading AR LoRA from {ar_lora_path} (scale: {ar_scale})...") 
                ar_data = load_ckpt(str(ar_lora_path), "cuda") 
                n_merged = merge_lora(bb, "self_attn", "mlp", ar_data["lora"], scale=ar_scale, dev="cuda") 
                print(f"✅ Merged {n_merged} AR LoRA linear layers.") 

            if nar_lora_path and nar_lora_path.exists(): 
                print(f"Loading NAR LoRA from {nar_lora_path} (scale: {nar_scale})...") 
                nar_data = load_ckpt(str(nar_lora_path), "cuda") 
                n_merged = merge_lora(bb, "nar_self_attn", "nar_mlp", nar_data["lora"], scale=nar_scale, dev="cuda") 
                
                # Apply full replacement weights for VAE <-> LLM crossover 
                top_model.vae2llm.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["vae2llm"].items()}) 
                top_model.llm2vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["llm2vae"].items()}) 
                print(f"✅ Merged {n_merged} NAR LoRA linear layers & custom VAE/LLM crossovers.") 

        print("\n>>> Generating audio...") 
        cover = pipe( 
            style=style, 
            lyrics=lyrics_content, 
            abc=abc_content, 
            cot=cot, 
            seed=seed 
        ) 
        
        cover.save_artifacts(output_dir) 
        print(f"\n✅ Cover track saved to: {output_dir}/audio.flac") 
        print(f"✅ Parameters saved to: {output_dir}/parameters.yaml")

if __name__ == "__main__": 
    main()
