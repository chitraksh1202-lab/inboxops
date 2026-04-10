"""
server/app.py - OpenEnv-compatible server entry point for InboxOps.
"""

import os
import sys
import uuid

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from fastapi import FastAPI, Request, HTTPException
from env import InboxOpsEnv

app = FastAPI(title="InboxOps OpenEnv Server")

_sessions: dict = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    task = body.get("task") or body.get("task_id") or "email_triage"
    try:
        env = InboxOpsEnv()
        state = env.reset(task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    session_id = str(uuid.uuid4())
    _sessions[session_id] = env
    return {"session_id": session_id, "state": state}


@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_id = body.get("session_id")
    action = body.get("action")
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        state, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"state": state, "reward": reward, "done": done, "info": info}


@app.get("/grade")
async def grade(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"score": env.grade(), "summary": env.summary()}


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
