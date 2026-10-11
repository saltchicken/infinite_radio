import argparse 
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
    parser = argparse.ArgumentParser(description="Generate a song cover using an ABC file and optional LoRAs.") 
    parser.add_argument("--abc", type=Path, required=True, help="Path to the .abc sheet music file") 
    parser.add_argument("--style", type=str, required=True, help="Style description string") 
    parser.add_argument("--lyrics", type=str, required=True, help="Lyrics text, or path to a text file containing lyrics") 
    parser.add_argument("--output", type=Path, default=Path("outputs/cover"), help="Output directory") 
    
    # LoRA arguments 
    parser.add_argument("--ar-lora", type=Path, help="Path to the AR LoRA safetensors file") 
    parser.add_argument("--ar-scale", type=float, default=1.0, help="Scale for the AR LoRA") 
    parser.add_argument("--nar-lora", type=Path, help="Path to the NAR LoRA safetensors file") 
    parser.add_argument("--nar-scale", type=float, default=1.0, help="Scale for the NAR LoRA") 
    
    # Generation arguments 
    parser.add_argument("--seed", type=int, default=42, help="Seed for generation") 
    parser.add_argument("--cot", type=str, default="full", help="Chain-of-thought mode") 
    args = parser.parse_args() 

    args.output.mkdir(parents=True, exist_ok=True) 

    # 1. Load ABC content 
    if not args.abc.exists(): 
        print(f"❌ Error: ABC file not found at {args.abc}") 
        return 
    abc_content = args.abc.read_text(encoding="utf-8") 

    # 2. Load Lyrics (allows passing a text file or an inline string) 
    lyrics_path = Path(args.lyrics) 
    if lyrics_path.is_file(): 
        lyrics_content = lyrics_path.read_text(encoding="utf-8") 
    elif args.lyrics.endswith(".txt"): 
        # If it looks like a text file but doesn't exist, throw an error 
        print(f"❌ Error: Lyrics file '{args.lyrics}' not found. Check your spelling.") 
        return 
    else: 
        # Otherwise, treat the input as literal lyric text 
        lyrics_content = args.lyrics 

    # 3. Print out all parameters being used 
    print(f"\n======================================") 
    print(f"🎸 Generating Cover") 
    print(f"======================================") 
    print(f"ABC Source : {args.abc}") 
    print(f"Output Dir : {args.output}") 
    print(f"Seed       : {args.seed}") 
    print(f"CoT Mode   : {args.cot}") 
    
    if args.ar_lora: 
        print(f"AR LoRA    : {args.ar_lora} (Scale: {args.ar_scale})") 
    if args.nar_lora: 
        print(f"NAR LoRA   : {args.nar_lora} (Scale: {args.nar_scale})") 
        
    print(f"\n--- Style ---") 
    print(args.style) 
    
    print(f"\n--- Lyrics ---") 
    print(lyrics_content.strip()) 
    print(f"======================================\n") 

    print("Loading YuE2 Base Model...") 
    with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", device="cuda") as pipe: 
        
        # --- Inject LoRAs if specified --- 
        if args.ar_lora or args.nar_lora: 
            print("\n>>> Injecting LoRAs into the model...") 
            top_model = pipe._load_model() 
            bb = top_model.model 
            
            if args.ar_lora and args.ar_lora.exists(): 
                print(f"Loading AR LoRA from {args.ar_lora} (scale: {args.ar_scale})...") 
                ar_data = load_ckpt(str(args.ar_lora), "cuda") 
                n_merged = merge_lora(bb, "self_attn", "mlp", ar_data["lora"], scale=args.ar_scale, dev="cuda") 
                print(f"✅ Merged {n_merged} AR LoRA linear layers.") 

            if args.nar_lora and args.nar_lora.exists(): 
                print(f"Loading NAR LoRA from {args.nar_lora} (scale: {args.nar_scale})...") 
                nar_data = load_ckpt(str(args.nar_lora), "cuda") 
                n_merged = merge_lora(bb, "nar_self_attn", "nar_mlp", nar_data["lora"], scale=args.nar_scale, dev="cuda") 
                
                # Apply full replacement weights for VAE <-> LLM crossover 
                top_model.vae2llm.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["vae2llm"].items()}) 
                top_model.llm2vae.load_state_dict({k: v.to(torch.bfloat16) for k, v in nar_data["io"]["llm2vae"].items()}) 
                print(f"✅ Merged {n_merged} NAR LoRA linear layers & custom VAE/LLM crossovers.") 

        print("\n>>> Generating audio...") 
        cover = pipe( 
            style=args.style, 
            lyrics=lyrics_content, 
            abc=abc_content, 
            cot=args.cot, 
            seed=args.seed 
        ) 
        
        cover.save_artifacts(args.output) 
        print(f"\n✅ Cover track saved to: {args.output}/audio.flac") 

if __name__ == "__main__": 
    main()
