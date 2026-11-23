"""End-to-end: send a question, verify LLM answer + audio + frames."""
import asyncio
import json
import websockets

async def main():
    uri = "ws://localhost:8080/ws"
    async with websockets.connect(uri, max_size=20 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "text": "Расскажи в двух предложениях, зачем ты построил Петербург?",
            "mode": "ask",
        }))
        frames = 0
        audio_bytes = 0
        answer = None
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
            except asyncio.TimeoutError:
                print("TIMEOUT")
                break
            if isinstance(msg, bytes):
                if audio_bytes == 0:
                    audio_bytes = len(msg)
                else:
                    frames += 1
            else:
                obj = json.loads(msg)
                t = obj.get("type")
                if t == "answer":
                    answer = obj.get("text")
                    print(f"ANSWER: {answer}")
                elif t == "status":
                    print(f"status: {obj.get('message')}")
                elif t == "stream_end":
                    print(f"DONE frames={obj.get('frames_sent')} elapsed={obj.get('elapsed')}s")
                    break
                elif t == "error":
                    print(f"ERROR: {obj.get('message')}")
                    break
        print(f"summary: answer={'yes' if answer else 'NO'}  audio_bytes={audio_bytes}  frames={frames}")

asyncio.run(main())
