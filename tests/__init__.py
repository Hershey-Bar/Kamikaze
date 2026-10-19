import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_KAMIKAZE = _ROOT / "kamikaze"
_TESTS = _ROOT / "tests"
for _p in (str(_ROOT), str(_KAMIKAZE), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
