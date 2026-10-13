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
    parser.add_argument("-o", "--output", default=".", help="Base output directory (default: current directory)")
    
    args = parser.parse_args()
    
    # 1. Gather all tasks
    tasks = []
    if args.batch:
        batch_path = Path(args.batch)
        filename_stem = batch_path.stem
        
        # Count the occurrences to catch unexpected multiple dashes
        dash_count = filename_stem.count(" - ")
        
        if dash_count > 1:
            print(f"Error: Unexpected filename format. Found multiple ' - ' in '{filename_stem}.csv'. Expected '<Spreadsheet> - <GameName>.csv'.")
            sys.exit(1)
        elif dash_count == 1:
            # Split "<Spreadsheet> - <GameName>" based on Google Sheets download format
            spreadsheet, game_name = filename_stem.split(" - ")
            # Use Spreadsheet/GameName as the path
            batch_subfolder = Path(spreadsheet.strip()) / game_name.strip()
        else:
            # Fallback to the whole file name if no dash is present
            batch_subfolder = Path(filename_stem.strip())

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
                            
                    tasks.append({"url": url, "name": name, "subfolder": batch_subfolder})
    elif args.url and args.name:
        # Fallback to provided args.subfolder for single runs
        tasks.append({"url": args.url, "name": args.name, "subfolder": Path(args.subfolder) if args.subfolder else None})
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
    
    # Consolidate output to the base output directory
    base_dir = Path(args.output).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            try:
                model.transcribe(
                    str(audio_path),
                    output_dir=str(run_dir),
                    melody_only=run['melody_only']
                )
            except RuntimeError as e:
                if "cannot be represented on the decoded subbeat grid" in str(e):
                    print(f"⚠️ Skipping transcription: SheetSage couldn't quantize the timing for sheet music. Error: {e}")
                    continue  # Skip to the next run or track without attempting to rename files
                else:
                    raise e
            except Exception as e:
                print(f"❌ An unexpected error occurred during transcription: {e}")
                continue  # Skip to the next run or track without attempting to rename files
            
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
