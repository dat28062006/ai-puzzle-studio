"""Regenerate the real sample level, report, animation, and export bundle."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parents[1] / "work" / "dependencies"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT))

from puzzle.art import load_assets
from puzzle.engine import PRESETS
from puzzle.generator import generate
from puzzle.rendering import png_bytes, render_board, replay_gif
from puzzle.reports import build_report, export_bundle, json_bytes
from puzzle.solver import replay, solve
from puzzle.experiments import compare_agents
from puzzle.q_learning import load_model, rollout
from puzzle.engine import from_ascii


def main():
    destination = ROOT / "examples"
    destination.mkdir(exist_ok=True)
    generated = generate(42)
    level, result = generated.level, generated.verification
    replay(level, result)
    assets = load_assets()
    (destination / "level.json").write_bytes(json_bytes(level.to_dict()))
    (destination / "report.json").write_bytes(json_bytes(build_report(level, level.initial, result)))
    (destination / "preview.png").write_bytes(png_bytes(render_board(level, assets=assets)))
    (destination / "solution.gif").write_bytes(replay_gif(level, result, assets))
    (destination / "puzzle-lab-export.zip").write_bytes(export_bundle(level, level.initial, result, assets=assets))
    rows = []
    for preset in PRESETS:
        outcome = solve(preset)
        replay(preset, outcome)
        rows.append({"level": preset.to_dict(), "search": outcome.to_dict()})
    (destination / "benchmark.json").write_bytes(json_bytes(rows))
    users_level = generate(60, template="Lối vòng").level
    comparisons = compare_agents(users_level)
    (destination / "agent-comparison-seed60.json").write_bytes(json_bytes({"level": users_level.to_dict(), "results": comparisons}))
    model = load_model(users_level)
    if model:
        outcome = rollout(users_level, model)
        (destination / "rl-training-seed60.json").write_bytes(json_bytes({"training": model.training, "history": model.history}))
        if outcome.solved:
            (destination / "solution-rl-seed60.gif").write_bytes(replay_gif(users_level, outcome, assets))
    tradeoff = from_ascii("#######\n# #  @#\n# $   #\n#     #\n#. $  #\n# .   #\n#######", id="objective-tradeoff", title="Ưu tiên bước đi hay lượt đẩy?")
    (destination / "objective-tradeoff.json").write_bytes(json_bytes(tradeoff.to_dict()))
    print(f"Saved verified examples in {destination}")
    print(f"Seed 42: {len(result.actions)} moves, {result.pushes} pushes, status={result.status}")


if __name__ == "__main__":
    main()
