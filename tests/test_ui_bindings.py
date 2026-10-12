"""Static checks on the Training tab JavaScript - no browser needed."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parent.parent / "app" / "static" / "js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("name", sorted(p.name for p in JS.glob("*.js")))
def test_js_parses(name):
    r = subprocess.run(["node", "--check", str(JS / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_default_setting_is_bound_to_a_field():
    s = (JS / "train.js").read_text(encoding="utf-8")
    a = s.index("function defaultState") if "function defaultState" in s else s.index("const state")
    seg = s[a:a + 9000]
    end = seg.find("\n  function ")
    block = seg[:end if end > 0 else len(seg)]
    keys = set(re.findall(r"\b([a-z][a-z_0-9]+):", block))
    rest = s[a + len(block):]
    unbound = [k for k in sorted(keys) if not re.search(r"\." + k + r"\b", rest)]
    assert not unbound, f"settings with no UI field: {unbound}"


def test_all_section_builders_are_defined_and_used():
    s = (JS / "train.js").read_text(encoding="utf-8")
    defined = set(re.findall(r"function (build\w+Section)\(", s))
    used = set(re.findall(r"\b(build\w+Section)\(\)", s))
    assert used <= defined, f"called but not defined: {used - defined}"
    assert {"buildGeneralSection", "buildModelSection", "buildNetworkSection",
            "buildOptimizerSection", "buildSavingSection"} <= defined
