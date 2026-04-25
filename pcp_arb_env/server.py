"""
OpenEnv Server — FastAPI wrapper for PCPArbEnv.
Exposes standard endpoints for external agents/graders.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from pcp_arb_env.environment import PCPArbEnv
from data.feeds.mock_feed import MockFeed
from mcp_servers.mcp_client import MCPClient
from config.settings import get_settings

app = FastAPI(title="PCP Arb OpenEnv Server")

# Global environment instance
_env: Optional[PCPArbEnv] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]

@app.on_event("startup")
async def startup():
    global _env
    settings = get_settings()
    # Initialize with default mock feed for the server
    feed = MockFeed(underlyings=["NIFTY"])
    mcp = MCPClient()
    _env = PCPArbEnv(feed=feed, mcp_client=mcp)

@app.post("/reset")
async def reset():
    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    obs = _env.reset()
    return {"observation": obs}

@app.post("/step")
async def step(req: StepRequest):
    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    if _env.done:
        return {"error": "Episode finished. Please reset."}
    
    result = _env.step(req.action)
    return {
        "observation": result.observation,
        "reward": result.reward.total,
        "done": result.done,
        "info": result.info
    }

@app.get("/state")
async def get_state():
    if _env is None:
        return {"status": "uninitialized"}
    return _env.state()

@app.get("/health")
async def health():
    return {"status": "ok", "environment": "PCPArbEnv", "done": _env.done if _env else False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
