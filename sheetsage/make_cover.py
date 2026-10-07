from pathlib import Path
from yue2 import YuE2Pipeline

with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda") as pipe:
    cover = pipe(
        style="synthwave",
        lyrics="[Intro]\n[Verse]\n[Chorus]\n[Verse]\n[Outro]",
        abc=Path("cover-score/score.abc").read_text(encoding="utf-8"),
        cot="full",
        seed=42,
    )
    cover.save_artifacts("outputs/cover-with-chords/1")
