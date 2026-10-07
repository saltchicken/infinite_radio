import torch
from transformers import AutoModel

# Load the SheetSage2 model
model = AutoModel.from_pretrained("m-a-p/SheetSage2", trust_remote_code=True)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Transcribe the audio file
# melody_only=True is required for YuE2 covers to omit chord symbols
result = model.transcribe("chrono_trigger-wind_scene.mp3",
                          output_dir="cover-score",
                          melody_only=True)

print("Success! ABC score saved to cover-score/score.abc")
