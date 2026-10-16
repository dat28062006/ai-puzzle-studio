"""Run using installed dependencies, or this Codex workspace's isolated bundle."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT.parents[1] / "work" / "dependencies"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from streamlit.web import cli
    sys.argv = ["streamlit", "run", str(ROOT / "app.py"), "--global.developmentMode=false", "--server.address=127.0.0.1",
                "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false", *sys.argv[1:]]
    raise SystemExit(cli.main())
