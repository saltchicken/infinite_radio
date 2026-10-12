#!/usr/bin/env python3
import argparse
import sys
import subprocess
import torch
from pathlib import Path
from transformers import AutoModel

def parse_duration(duration_str):
    """Converts MM:SS or HH:MM:SS to total seconds. Falls back to float."""
    duration_str = duration_str.strip()
    if not duration_str:
        return 0.0
        
    if ':' in duration_str:
        parts = duration_str.split(':')
        seconds = 0
        # If MM:SS
        if len(parts) == 2:
            seconds = int(parts[0]) * 60 + int(parts[1])
        # If HH:MM:SS
        elif len(parts) == 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return float(seconds)
        
    # Fallback if it's already just seconds
    return float(duration_str)

def download_audio(url, name, output_dir):
    print(f"\n==================================================")
    print(f"⬇️  Downloading: {name}")
    print(f"==================================================")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = f"{name}.%(ext)s"
    
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "--cookies-from-browser", "firefox",
        "-P", str(output_dir),
        "-o", output_template,
        url
    ]
    
    # Run yt-dlp using the system installation
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: yt-dlp failed for {url}")
        return None
        
    return output_dir / f"{name}.mp3"

def main():
    parser = argparse.ArgumentParser(description="Download and transcribe YouTube audio to ABC notation.")
    parser.add_argument("-b", "--batch", help="Path to a CSV file (URL, Name, Duration)")
    parser.add_argument("-u", "--url", help="Single YouTube URL")
    parser.add_argument("-n", "--name", help="Single Song Name")
    parser.add_argument("-s", "--subfolder", default="", help="Subfolder to save the audio/scores in (optional)")
    
    args = parser.parse_args()
    
    # 1. Gather all tasks
    tasks = []
    if args.batch:
        with open(args.batch, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    url = parts[0]
                    name = parts[1]
                    
                    # Check duration if provided in the 3rd column
                    if len(parts) > 2 and parts[2]:
                        try:
                            duration = parse_duration(parts[2])
                            if duration < 30:
                                print(f"⏭️ Skipping '{name}': Duration ({duration}s) is less than 30 seconds.")
                                continue
                        except ValueError:
                            print(f"⚠️ Warning: Could not parse duration '{parts[2]}' for '{name}'. Proceeding anyway.")
                            
                    tasks.append({"url": url, "name": name, "subfolder": args.subfolder})
    elif args.url and args.name:
        tasks.append({"url": args.url, "name": args.name, "subfolder": args.subfolder})
    else:
        print("Error: You must provide either a --batch CSV or both --url and --name.")
        sys.exit(1)
        
    if not tasks:
        print("No tasks found to process.")
        sys.exit(0)
        
    # 2. Load the model ONCE
    print("\n==================================================")
    print("🧠 Loading SheetSage2 Model...")
    print("==================================================")
    model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    
    # Consolidate output to the current working directory
    base_dir = Path.cwd()
    
    runs = [
        {"melody_only": True, "subfolder": "melody", "suffix": "-melody"},
        {"melody_only": False, "subfolder": "full", "suffix": "-full"}
    ]
    
    # 3. Interleave Downloads and Transcriptions
    for task in tasks:
        subfolder = task["subfolder"]
        output_dir = base_dir / subfolder if subfolder else base_dir
        
        # --- DOWNLOAD ---
        audio_path = download_audio(task["url"], task["name"], output_dir)
        if not audio_path or not audio_path.exists():
            print(f"Skipping transcription for {task['name']} due to download failure.")
            continue
            
        # --- TRANSCRIBE ---
        print(f"\n--- Transcribing: {task['name']} ---")
        target_dir = output_dir / f"{audio_path.stem}-score"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for run in runs:
            run_dir = target_dir / run["subfolder"]
            run_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Generating (melody_only={run['melody_only']}) to '{run_dir}'...")
            model.transcribe(
                str(audio_path),
                output_dir=str(run_dir),
                melody_only=run['melody_only']
            )
            
            default_output = run_dir / "score.abc"
            final_output = run_dir / f"{audio_path.stem}{run['suffix']}.abc"
            
            if default_output.exists():
                if final_output.exists() and default_output != final_output:
                    final_output.unlink()
                if default_output != final_output:
                    default_output.rename(final_output)
                print(f"Saved ABC file: {final_output}")
            else:
                print(f"Error: Expected output {default_output} was not generated.")
                
    print(f"\n✅ Success! All {len(tasks)} tasks complete.")

if __name__ == "__main__":
    sys.exit(main())
