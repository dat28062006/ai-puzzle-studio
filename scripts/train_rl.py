"""Train the same RL agent as the UI, without needing a browser."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parents[1] / "work" / "dependencies"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT))

import argparse
from puzzle.generator import TEMPLATES, generate
from puzzle.q_learning import rollout, save_model, train_q

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=60, help="Level generation seed")
parser.add_argument("--template", choices=list(TEMPLATES), default="Lối vòng")
parser.add_argument("--boxes", type=int, choices=[1, 2], default=2)
parser.add_argument("--episodes", type=int, default=50000)
parser.add_argument("--training-seed", type=int, default=42)
parser.add_argument("--budget", type=float, default=20)
args = parser.parse_args()
level = generate(args.seed, template=args.template, boxes=args.boxes).level
model = train_q(level, episodes=args.episodes, seed=args.training_seed, budget=args.budget,
                on_progress=lambda episode, total, row: print(f"{episode}/{total}: reward={row['mean_reward']:.2f}, greedy_solved={row['greedy_root_solved']}")
                if episode % 5000 == 0 else None)
path = save_model(level, model)
result = rollout(level, model)
print(f"Checkpoint: {path}")
print(f"Policy: {result.status}; moves={len(result.actions)}, pushes={result.pushes}")
