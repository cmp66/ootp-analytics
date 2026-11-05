"""The viewer has to fit on a small server (about 1 GB free next to other services). Opening every page in one process must stay
under a memory budget, must not load the model libraries, and must leave the shared tables unchanged. A new screen that loads
its own copy of the data, or changes the shared frame, fails here."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from util.update_config import load_update_config
from util import paths
from util.seasons import available_seasons

ROOT = Path(__file__).resolve().parent.parent
BUDGET_MB = 500  # the whole process, one league, every page; the server has about 940 MB free for it

SCRIPT = """
import json, sys
import pandas as pd
from streamlit.testing.v1 import AppTest
from viewer import data_cache
from viewer.screens import org_view

pages = [target for target, *_ in org_view.PAGES.values()]
for target in pages:
    app = AppTest.from_file("viewer/" + target, default_timeout=240).run()
    if app.exception:
        raise SystemExit(f"{target}: {app.exception[0].value}")
from util.update_config import load_update_config
save = next(iter(load_update_config()))
from util.seasons import available_seasons
from viewer import queries
season = available_seasons(save)[-1]
# the peak of this process (VmHWM; ru_maxrss would include the parent's), before the comparison below loads a second copy
peak = int(open("/proc/self/status").read().split("VmHWM:")[1].split()[0]) // 1024
fresh = data_cache.shrink(queries.load_outlook(save, season))
pd.testing.assert_frame_equal(data_cache.outlook(save, season), fresh)  # no page changed the shared frame
print(json.dumps({
    "peak_mb": peak,
    "heavy": [m for m in ("torch", "sklearn", "shap") if m in sys.modules],
}))
"""


def has_outlook_files() -> bool:
    save = next(iter(load_update_config()))
    seasons = available_seasons(save)
    return bool(seasons) and paths.output_file(save, seasons[-1], "outlook-slots").exists()


@pytest.mark.skipif(not has_outlook_files(), reason="needs the outlook files of scripts/update_all.py")
def test_every_page_fits_the_memory_budget_without_the_model_libraries_and_leaves_the_data_alone():
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT], cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": "."},
        capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-2000:] + done.stdout[-500:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["heavy"] == [], f"the viewer loaded {result['heavy']}"
    assert result["peak_mb"] <= BUDGET_MB, f"peak memory {result['peak_mb']} MB > {BUDGET_MB} MB"
    print(result)


def test_the_mmap_threshold_is_fixed_on_linux_and_the_call_is_harmless_elsewhere():
    from viewer import data_cache

    assert data_cache.fix_mmap_threshold() == sys.platform.startswith("linux")
