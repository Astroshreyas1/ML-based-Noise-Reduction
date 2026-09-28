"""MCP server for the hub's situation picture (stdio). Any MCP client -- a local model via an MCP client,
Claude Desktop, or Laya's own MCP tools side by side -- can read the picture and place operator markers.
It talks to the running hub's map server over HTTP, so the hot path never waits on MCP.

    python -m hubsuite.mcp_server            (hub running with its map server on 127.0.0.1:8770)

Tools: get_situation, list_contacts, recent_transmissions, add_marker, grid_to_latlon.
Read-mostly by design: the only write is an operator marker, labelled source=operator.
"""
from __future__ import annotations

import json
import os
import urllib.request

from mcp.server.fastmcp import FastMCP

from .radiotext import grid_to_latlon as _g2ll

HUB = os.environ.get("HUB_URL", "http://127.0.0.1:8770")
mcp = FastMCP("radio-hub-situation")


def _get(path: str) -> dict:
    with urllib.request.urlopen(HUB + path, timeout=5) as r:
        return json.loads(r.read())


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(HUB + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


@mcp.tool()
def get_situation() -> dict:
    """All markers (friendly, enemy, casualty, fire missions, check fire) with grid, source transmission id,
    confidence and whether the grid is confirmed."""
    s = _get("/api/state")
    return {"square": s["square"], "markers": s["markers"]}


@mcp.tool()
def list_contacts(kind: str = "enemy") -> list[dict]:
    """Markers of one kind: enemy | friendly | casualty | fire_mission | check_fire | note."""
    return [m for m in _get("/api/state")["markers"] if m["kind"] == kind]


@mcp.tool()
def recent_transmissions(n: int = 20) -> list[dict]:
    """The last n transcribed transmissions with their rule extraction and System-1 decisions."""
    return _get("/api/state")["transmissions"][-n:]


@mcp.tool()
def add_marker(grid: str, label: str, kind: str = "note") -> dict:
    """Place an OPERATOR marker at a grid (4-10 digits inside the configured 100 km square)."""
    return _post("/api/marker", {"grid": grid, "label": label, "kind": kind})


@mcp.tool()
def grid_to_latlon(grid: str, square: str = "33UVP") -> dict:
    """Convert a radio grid reference inside the 100 km square to latitude / longitude (+ uncertainty)."""
    lat, lon, half = _g2ll(grid, square)
    return {"lat": lat, "lon": lon, "half_size_m": half}


if __name__ == "__main__":
    mcp.run()
