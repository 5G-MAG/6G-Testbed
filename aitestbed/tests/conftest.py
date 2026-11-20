"""Make the testbed modules importable regardless of pytest's rootdir."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Test against the sibling netemu checkout, not whatever copy happens to be
# installed: an editable install can point at another worktree.
NETEMU_SRC = PROJECT_ROOT.parent / "netemu" / "src"
if NETEMU_SRC.is_dir():
    for name in [m for m in sys.modules if m == "netemu" or m.startswith("netemu.")]:
        del sys.modules[name]
    sys.path.insert(0, str(NETEMU_SRC))
