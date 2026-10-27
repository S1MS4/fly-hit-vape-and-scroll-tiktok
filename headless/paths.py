"""Project paths; importing this also makes brain/ (lif.py, world.py) importable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = ROOT / "brain"
BUILD = ROOT / "data" / "build"
PROFILE_DIR = ROOT / "data" / "tiktok-profile"
RUNS_DIR = ROOT / "data" / "tiktok-runs"

if str(BRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_DIR))
