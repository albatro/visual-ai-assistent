"""Test segment-based streaming protocol."""
import asyncio
import json
import time
import websockets

async def main():
    uri = "ws://localhost:8080/ws"
    async with websockets.connect(uri, max_size=20 * 1024 * 1024) as ws:
        t0 = time.time()
        await ws.send(json.dumps({
            "text": "Расскажи в трёх предложениях, зачем ты построил Петербург и почему именно на болоте?",
            "mode": "ask",
        }))

        first_audio_at = None
        first_frame_at = None
        segments = {}
        current_idx = None
        expecting_audio = False
        frames_remaining = 0
        total_frames = 0

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
            except asyncio.TimeoutError:
                print("TIMEOUT")
                break
            if isinstance(msg, bytes):
                if expecting_audio:
                    expecting_audio = False
                    if first_audio_at is None:
                        first_audio_at = time.time() - t0
                        print(f"  [t+{first_audio_at:.2f}s] seg {current_idx} AUDIO received {len(msg)}b")
                else:
                    frames_remaining -= 1
                    total_frames += 1
                    if first_frame_at is None:
                        first_frame_at = time.time() - t0
                        print(f"  [t+{first_frame_at:.2f}s] FIRST FRAME (seg {current_idx})")
            else:
                obj = json.loads(msg)
                t = obj.get("type")
                rel = time.time() - t0
                if t == "thinking":
                    print(f"[t+{rel:.2f}s] thinking")
                elif t == "segment":
                    current_idx = obj["idx"]
                    expecting_audio = True
                    frames_remaining = obj["frame_count"]
                    print(f"[t+{rel:.2f}s] segment {obj['idx']} text={obj['text'][:60]!r} frames={obj['frame_count']}")
                elif t == "segment_end":
                    print(f"[t+{rel:.2f}s] segment_end {obj['idx']}")
                elif t == "answer_complete":
                    print(f"[t+{rel:.2f}s] DONE total elapsed={obj['elapsed']}s")
                    break
                elif t == "error":
                    print(f"[t+{rel:.2f}s] ERROR: {obj.get('message')}")
                    break

        print(f"\n=== SUMMARY ===")
        print(f"first audio:  +{first_audio_at:.2f}s" if first_audio_at else "no audio")
        print(f"first frame:  +{first_frame_at:.2f}s" if first_frame_at else "no frames")
        print(f"total frames: {total_frames}")

asyncio.run(main())
