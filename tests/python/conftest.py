import sys
from pathlib import Path


def pytest_sessionstart(session):
    root = Path(__file__).resolve().parents[2]
    pkg_path = root / "codex_pod_agent" / "src" / "python"
    if "podagent" in sys.modules:
        del sys.modules["podagent"]
    sys.path.insert(0, str(pkg_path))
