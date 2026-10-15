import argparse
import json
import requests
import time
import zipfile
import io
from pathlib import Path

API_URL = "http://127.0.0.1:8000"

def main():
    parser = argparse.ArgumentParser(description="Recreate a 1:1 track from a metadata JSON file.")
    parser.add_argument("metadata_file", type=Path, help="Path to the metadata.json file")
    parser.add_argument("--output", type=Path, default=Path("recreated_track"), help="Folder to save the output")
    args = parser.parse_args()

    if not args.metadata_file.exists():
        print(f"Error: File not found - {args.metadata_file}")
        return

    # Load the metadata
    with open(args.metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Clean up local filesystem keys that the YuE2 generation pipeline doesn't accept
    for key in ["id", "_source_style", "_source_abc"]:
        metadata.pop(key, None)

    # Format the payload for the server
    # Note: Using your controller.json defaults for the required LoRA bindings
    payload = {
        "ar_lora": "loras/ar_lora_inst_v3abc.safetensors",
        "nar_lora": "loras/nar_lora_joint_v4.safetensors",
        "ar_scale": 0.5,
        "nar_scale": 1.0,
        "yue_kwargs": metadata
    }

    print(f"Submitting task to {API_URL}...")
    submit_resp = requests.post(f"{API_URL}/generate", json=payload)
    
    if not submit_resp.ok:
        print("Submission failed:", submit_resp.text)
        return
        
    task_id = submit_resp.json()["task_id"]
    print(f"Task ID: {task_id}")
    print("Generating (this may take a while)...")

    # Poll for completion
    while True:
        status_resp = requests.get(f"{API_URL}/status/{task_id}").json()
        status = status_resp["status"]
        
        if status == "completed":
            print("Done! Downloading artifacts...")
            break
        elif status == "failed":
            print("Generation failed:", status_resp.get("error", "Unknown server error"))
            return
            
        time.sleep(3)

    # Download and extract the zip stream in memory
    args.output.mkdir(parents=True, exist_ok=True)
    zip_content = requests.get(f"{API_URL}/download/{task_id}").content
    
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        zf.extractall(args.output)
        
    print(f"Success! Track saved to {args.output.resolve()}")

if __name__ == "__main__":
    main()
