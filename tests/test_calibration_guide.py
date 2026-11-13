# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""The example in docs/guides/calibration.md runs as documented."""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_guide_example_runs(capsys):
    text = (ROOT / "docs" / "guides" / "calibration.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert len(blocks) == 1
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        exec(compile(blocks[0], "calibration.md", "exec"), {"__name__": "guide"})
    finally:
        os.chdir(cwd)
    assert "mm" in capsys.readouterr().out
