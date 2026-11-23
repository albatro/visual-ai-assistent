"""One-off runner: lip-sync the kling reference video with the male voice clip."""

import cv2
from pathlib import Path
from musetalk_wrapper import MuseTalkEngine

REF_VIDEO = "kling_peter_cropped.mp4"
AUDIO = "test_voice_male.wav"
OUTPUT = "kling_peter_lipsync_raw.mp4"
FPS = 24

engine = MuseTalkEngine(fps=FPS)
engine.load_models()

avatar = engine.prepare_avatar("kling_peter_cropped", REF_VIDEO)
frames = engine.generate_frames(avatar, AUDIO)

if not frames:
    raise SystemExit("No frames generated")

h, w = frames[0].shape[:2]
writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
for f in frames:
    writer.write(f)
writer.release()
print(f"Wrote {len(frames)} frames -> {OUTPUT}")
