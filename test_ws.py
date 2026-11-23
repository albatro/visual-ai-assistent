"""End-to-end test: connect to streaming_server, send text, verify frames arrive."""
import asyncio
import json
import websockets

async def main():
    uri = "ws://localhost:8080/ws"
    async with websockets.connect(uri, max_size=20 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"text": "Привет, я Пётр Первый, основатель Петербурга."}))
        frames = 0
        audio_bytes = 0
        statuses = []
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, bytes):
                if audio_bytes == 0:
                    audio_bytes = len(msg)
                else:
                    frames += 1
            else:
                obj = json.loads(msg)
                statuses.append(obj.get("type"))
                if obj.get("type") == "stream_end":
                    print(f"DONE  frames_sent={obj.get('frames_sent')} elapsed={obj.get('elapsed')}s")
                    break
                if obj.get("type") == "error":
                    print(f"ERROR {obj.get('message')}")
                    break
        print(f"frames={frames}  audio_bytes={audio_bytes}  status_events={statuses}")

asyncio.run(main())
