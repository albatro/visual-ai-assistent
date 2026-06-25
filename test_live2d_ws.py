"""Smoke test for branches.live2d_branch /ws — sends one short say-mode
request and asserts the protocol invariants the browser client relies on.

Uses aiohttp's WS client. (The websockets PyPI package v16.0 fails the
handshake against our uvicorn — curl and browsers work fine, so it's a
client-side incompatibility, not a server bug.)
"""
import asyncio
import json
import sys

import aiohttp


async def main() -> int:
    seen = {"stack": 0, "segment": 0, "rig": 0, "segment_end": 0,
            "answer_complete": 0, "audio_bytes": 0, "error": 0}
    rig_samples: list[dict] = []

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            "ws://localhost:8082/ws", timeout=aiohttp.ClientWSTimeout(ws_close=30),
        ) as ws:
            await ws.send_str(json.dumps({
                "text": "Здравствуй, мир! Я говорю через тестовый канал.",
                "mode": "say", "tts": "edge",
            }))

            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=20)
                except asyncio.TimeoutError:
                    print("TIMEOUT waiting for messages")
                    return 2

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    t = data.get("type", "?")
                    seen[t] = seen.get(t, 0) + 1
                    if t == "rig":
                        rig_samples.append(data)
                        if len(rig_samples) <= 3 or len(rig_samples) % 20 == 0:
                            print(f"  <-rig t={data['t']:.2f}s "
                                  f"params={list(data['params'].keys())}")
                    else:
                        print(f"  <-{t}: {data}")
                    if t == "answer_complete":
                        break
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    seen["audio_bytes"] += len(msg.data)
                    print(f"  <-audio blob: {len(msg.data)} bytes")
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    print(f"  <-{msg.type}: {msg.data}")
                    break

    print("\n-- summary --")
    for k, v in seen.items():
        print(f"  {k:18s} {v}")
    print(f"  rig msgs:        {len(rig_samples)}")
    if rig_samples:
        all_params: set[str] = set()
        for r in rig_samples:
            all_params.update(r["params"].keys())
        print(f"  rig params seen: {sorted(all_params)}")

    fail = []
    if seen["stack"] != 1:             fail.append("stack != 1")
    if seen["segment"] < 1:            fail.append("no segment")
    if seen["audio_bytes"] == 0:       fail.append("no audio bytes")
    if seen["rig"] == 0:               fail.append("no rig updates")
    if seen["segment_end"] < 1:        fail.append("no segment_end")
    if seen["answer_complete"] != 1:   fail.append("answer_complete != 1")
    if seen["error"] != 0:             fail.append("got error msgs")
    if fail:
        print("\nFAIL:", ", ".join(fail))
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
