"""Organization overview page: one organization on current values, everything on one screen (plan 15)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from viewer.screens import org_view  # noqa: E402

org_view.render("org")
