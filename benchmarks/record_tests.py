"""Run every test suite and record the counts in benchmarks/results/tests.json.

The README, dashboard and case study quote the test count from this file, so
there is one number everywhere and it comes from a real run.

    .venv/bin/python benchmarks/record_tests.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "results" / "tests.json"
DASHBOARD_COPY = ROOT / "dashboard" / "lib" / "measured" / "tests.json"


def run_cargo() -> dict:
    proc = subprocess.run(["cargo", "test", "--quiet"], cwd=ROOT / "pageforge-rs", capture_output=True, text=True)
    passed = failed = 0
    for m in re.finditer(r"test result: \w+\. (\d+) passed; (\d+) failed", proc.stdout):
        passed += int(m.group(1))
        failed += int(m.group(2))
    return {"passed": passed, "failed": failed, "exit_code": proc.returncode}


def run_pytest() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml_path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        suite = ET.parse(xml_path).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
        assert suite is not None
        total = int(suite.get("tests", 0))
        failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
        by_file: dict[str, int] = {}
        for case in suite.iter("testcase"):
            name = case.get("classname", "").split(".")[-1]
            by_file[name] = by_file.get(name, 0) + 1
    return {
        "collected": total,
        "passed": total - failed - skipped,
        "failed": failed,
        "skipped": skipped,
        "by_file": by_file,
        "exit_code": proc.returncode,
    }


def main() -> int:
    import torch

    rust, python = run_cargo(), run_pytest()
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rust": rust,
        "python": python,
        "total_passed": rust["passed"] + python["passed"],
        "total": rust["passed"] + rust["failed"] + python["collected"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # The dashboard deploys from dashboard/, so it gets its own copy.
    DASHBOARD_COPY.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_COPY.write_text(json.dumps(result, indent=2) + "\n")
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    ok = rust["exit_code"] == 0 and python["exit_code"] == 0 and python["skipped"] == 0
    if python["skipped"]:
        print("note: tests were skipped (no GPU?); counts are not a full run", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
