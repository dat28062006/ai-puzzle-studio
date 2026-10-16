from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import zipfile

from .art import ASSET_KEYS, asset_manifest
from .engine import Level, State, won
from .rendering import png_bytes, render_board, replay_gif
from .solver import SearchResult


def json_bytes(value: dict | list) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def build_report(level: Level, state: State, result: SearchResult | None,
                 moves: int = 0, pushes: int = 0, overrides: dict | None = None,
                 skin: str = "GenAI") -> dict:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": level.to_dict(),
        "play": {"moves": moves, "pushes": pushes, "won": won(level, state), "current_state": state.to_dict()},
        "search": result.to_dict() if result else None,
        "difficulty": {"status": "not_calibrated", "note": "Search metrics are not calibrated human difficulty."},
        "visuals": {"mode": skin, "manifest": asset_manifest(overrides)},
    }


def export_bundle(level: Level, state: State, result: SearchResult | None, *,
                  assets: dict, moves: int = 0, pushes: int = 0,
                  overrides: dict | None = None, skin: str = "GenAI") -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("level.json", json_bytes(level.to_dict()))
        bundle.writestr("report.json", json_bytes(build_report(level, state, result, moves, pushes, overrides, skin)))
        bundle.writestr("preview.png", png_bytes(render_board(level, state, assets=assets)))
        bundle.writestr("assets/prompts.json", json_bytes(asset_manifest(overrides)))
        for key in ASSET_KEYS:
            if key in assets:
                bundle.writestr(f"assets/{key}.png", png_bytes(assets[key].resize((64, 64)) ))
        if result and result.solved:
            bundle.writestr("solution.gif", replay_gif(level, result, assets))
        bundle.writestr("README.txt", "AI Puzzle Studio\nlevel.json: original level, importable into the app.\n"
                        "report.json: current play state and real search/policy metrics; search.start_state identifies the origin.\n"
                        "preview.png: current board. solution.gif: replay from the search origin, when solved.\n"
                        "assets: normalized tiles and generation prompts/provenance.\n")
    return stream.getvalue()
