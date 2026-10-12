"""fp8 against k8v4 on every shipped lane, decided by the code cells.

The rule's form: the deciding cells are the 225-character `code` prompt and 36,000 characters of real
code, with the worst of the two deciding, and the non-code cells are disclosed rather than weighed.
The incumbent format is the one the lane ships and the record already holds its figures, so only the
alternative has to be measured - one run per lane, on the two deciding cells.

    python tools/bench/compare_kv_formats.py            # report
    python tools/bench/compare_kv_formats.py --lanes 3  # first three only
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "release"))

from profiles import PROFILES  # noqa: E402

RECORD = REPO / "tools" / "release" / "lane_figures.json"
CELLS = ("code", "long-code", "long-chinese")


def record_for(lane: str) -> dict:
    payload = json.loads(RECORD.read_text(encoding="utf-8"))
    return payload.get("lanes", {}).get(lane, {}).get("cells", {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lanes", type=int, default=0, help="limit to the first N lanes")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()

    lanes = [p for p in PROFILES if "vision" in p.get("file", "")]
    if args.lanes:
        lanes = lanes[:args.lanes]

    print(f"  {'lane':40s} {'shipped':>8} {'alt':>6} {'code (inc/alt)':>22} {'long-code':>18} "
          f"{'worst deciding':>16}")
    for profile in lanes:
        lane = profile["file"]
        incumbent = profile.get("kv_dtype", "?")
        alternative = "fp8" if incumbent != "fp8" else "k8v4"
        cells = record_for(lane)
        inc_code = cells.get("code", {}).get("decode_avg")
        inc_long = cells.get("long-code", {}).get("decode_avg")
        run = subprocess.run(
            [r"C:\vllm-env\Scripts\python.exe", "tools/bench/lane_fingerprint.py", "--file", lane,
             "--cell", "code", "--cell", "long-code", "--kv-dtype", alternative, "--no-write"],
            cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False)
        out = (run.stdout or "") + (run.stderr or "")
        alt_code = re.search(r"code\s+([\d.]+) tok/s", out)
        alt_long = re.search(r"long-code\s+([\d.]+) tok/s", out)
        if not (alt_code and alt_long) or inc_code is None or inc_long is None:
            # name the cause: a lane that refused to serve (rc=2) and a bad argument both land here,
            # and printing "no reading" alone is how this tool hid its own wrong flag for eight lanes
            first = next((line for line in out.splitlines() if line.strip()), "(no output)")
            print(f"  {lane:40s}  no reading (rc={run.returncode}): {first.strip()[:90]}")
            continue
        ac, al = float(alt_code.group(1)), float(alt_long.group(1))
        inc_worst = min(inc_code, inc_long)
        alt_worst = min(ac, al)
        verdict = f"{alternative} wins" if alt_worst > inc_worst else f"{incumbent} wins"
        print(f"  {lane:40s} {incumbent:>8} {alternative:>6} "
              f"{inc_code:>9.1f}/{ac:<9.1f} {inc_long:>8.1f}/{al:<8.1f} "
              f"{inc_worst:>7.1f}/{alt_worst:<7.1f} {verdict}")

    print("\n  The verdict compares each format's worst deciding cell, which is the rule's form."
          "\n  Margins inside this card's 10.3% session spread are not decisions: re-run before acting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
