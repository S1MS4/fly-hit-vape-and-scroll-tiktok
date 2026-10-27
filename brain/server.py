"""Brain server: runs the connectome simulation and streams it to the browser.

Run from the project root:  python brain/server.py
WebSocket /ws
  client -> server  binary: uint16 width, uint16 height, width*height uint8 luminance
                    text:   {"cmd": "puff" | "reset" | "pause" | "resume"}
  server -> client  text:   world state JSON every tick
                    binary: uint32 indices of neurons that spiked this tick
"""
import asyncio
import json
import struct
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lif import Brain
from world import World

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "data" / "build"
MODELS = ROOT / "models"
DIST = ROOT / "web" / "dist"
TICK_MS = 50          # biological time simulated per tick
MAX_SPIKES_SENT = 40000

app = FastAPI()
graph = np.load(BUILD / "graph.npz")
meta = json.loads((BUILD / "groups.json").read_text())
retina = np.load(BUILD / "retina.npz")
brain = Brain(graph["indptr"], graph["indices"], graph["counts"], graph["is_ach"])
steps_per_tick = int(round(TICK_MS / brain.p.dt))


def new_world():
    return World(meta["groups"], retina["idx"], retina["eye"], retina["u"], retina["v"])


@app.get("/api/meta")
def get_meta():
    return JSONResponse({
        "n": meta["n"], "edges": meta["edges"], "categories": meta["categories"],
        "front_leg_dn_types": meta["front_leg_dn_types"],
        "eliquid_orn_types": meta["eliquid_orn_types"],
        "group_sizes": {k: len(v) for k, v in meta["groups"].items()},
        "tick_ms": TICK_MS,
    })


@app.get("/api/somata")
def get_somata():
    return FileResponse(BUILD / "somata.bin", media_type="application/octet-stream")


def tick(world):
    idx, rates = world.sensory_inputs()
    brain.set_inputs(idx, rates)
    brain.ach_gain = world.ach_gain()
    counts = brain.run(steps_per_tick)
    spiked = counts.nonzero().squeeze(1)
    if spiked.numel() > MAX_SPIKES_SENT:
        spiked = spiked[torch.randperm(spiked.numel(), device=spiked.device)[:MAX_SPIKES_SENT]]
    state = world.update(counts.cpu().numpy(), TICK_MS / 1000)
    return state, spiked.to(torch.int32).cpu().numpy().tobytes()


# One brain, one viewer: a new connection takes over and the previous one is closed.
TAKEN_OVER = 4001
session = {"id": 0}
sim_lock = asyncio.Lock()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session["id"] += 1
    my_id = session["id"]
    world = new_world()
    paused = False
    reset = True

    async def receive():
        nonlocal paused, reset
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if msg.get("bytes"):
                data = msg["bytes"]
                w, h = struct.unpack_from("<HH", data)
                lum = np.frombuffer(data, dtype=np.uint8, offset=4, count=w * h)
                world.set_frame(lum.reshape(h, w).astype(np.float32) / 255.0)
            elif msg.get("text"):
                cmd = json.loads(msg["text"]).get("cmd")
                if cmd == "puff":
                    if not world.puff():
                        # A convulsing or slumped fly can't get the vape to its mouth. The
                        # page guards the button with cough_s_left/whitey_s_left, so this
                        # is a click that raced the tick one of them started in: say so,
                        # rather than dropping it silently while the browser animates it.
                        print(f"puff refused at t={world.t:.2f}s (cough {world.cough_t:.2f}s,"
                              f" whitey {world.whitey_t:.2f}s left)", flush=True)
                elif cmd == "reset":
                    reset = True
                elif cmd in ("pause", "resume"):
                    paused = cmd == "pause"

    receiver = asyncio.create_task(receive())
    try:
        while not receiver.done() and session["id"] == my_id:
            if paused:
                await asyncio.sleep(0.05)
                continue
            async with sim_lock:
                if reset:
                    world = new_world()
                    brain.reset()
                    reset = False
                state, spikes = await asyncio.to_thread(tick, world)
                state["bio_ms"] = round(brain.t_ms, 1)
            await ws.send_text(json.dumps(state))
            await ws.send_bytes(spikes)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receiver.cancel()
        if session["id"] != my_id:
            try:
                await ws.close(code=TAKEN_OVER)
            except RuntimeError:
                pass


# 3D models, and the built frontend (npm run build) so the desktop app needs no Vite.
if MODELS.is_dir():
    app.mount("/models", StaticFiles(directory=MODELS), name="models")
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
