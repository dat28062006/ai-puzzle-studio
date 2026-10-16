from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parents[1] / "work" / "dependencies"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT))
