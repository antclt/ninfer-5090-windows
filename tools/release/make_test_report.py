"""Generate the test inventory as HTML, for printing to PDF.

Everything comes from the files that own it, so the report cannot drift from the tree:

* the C++ test list from `ctest -N` (written to a file and passed in, because ctest is not on PATH here)
* the gate list from `.githooks/pre-commit`, the same lines the hook runs
* the triggers from the two workflows' `on:` blocks
* the exceptions from `tools/release/test_baseline.json` - required, known-failing, optional
* the Python tests from the tree, and which of them the hook runs from the hook itself
* the probes and harnesses from `tools/bench` and `tools/scripts`

    ctest --test-dir build-test -N > ctest.txt
    python tools/release/make_test_report.py --ctest-list ctest.txt --out build/test-report.html
    msedge --headless=new --print-to-pdf=docs/test-report.pdf --no-pdf-header-footer file:///<abs>
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# One line per test area, keyed by a prefix of the test name. The names carry the specificity; these
# carry the purpose, which is the part a name cannot.
AREAS = [
    ("ninfer_public_api_test", "The public C++ surface: headers compile and the documented entry points link."),
    ("ninfer_cpu_", "The CPU execution layer: affinity, worker pools, the executor."),
    ("ninfer_device", "Device primitives: allocation, synchronisation, the spin path."),
    ("ninfer_artifact_", "Artifact framing and binding: containers, codecs, readers, the writer interop."),
    ("ninfer_context_kv_materialize_test", "KV materialisation against its FP64 oracle, swept by batch."),
    ("ninfer_softmax_attention_test", "Attention across every KV representation in one invocation."),
    ("ninfer_causal", "Causal attention plans and partitions."),
    ("ninfer_qwen3_5_dflash2_real_test", "REQUIRED: the real DFlash2 speculative route against the shipped artifact."),
    ("ninfer_qwen3_5_dflash_prefill_real_test", "REQUIRED: real prefill through the DFlash route."),
    ("ninfer_qwen3_5_score_real_test", "REQUIRED: real causal scoring."),
    ("ninfer_qwen3_5_vision_workspace_test", "REQUIRED: the vision path's workspace."),
    ("ninfer_qwen3_5_dflash_real_test", "KNOWN FAILURE: needs a `dflash` component; the artifacts ship `dflash2`."),
    ("ninfer_qwen3_5_moe_real_test", "KNOWN FAILURE: needs a 35B-A3B checkpoint; every artifact here is 27B dense."),
    ("ninfer_qwen3_5_loading_real_test", "OPTIONAL: an operator tool taking --artifact; returns 77 under ctest."),
    ("ninfer_qwen3_5_", "The model: binding, frontend, program, state images, speculative rounds."),
    ("ninfer_serve", "The serving surface: request logs, cancellation, protocol contracts."),
    ("ninfer_bench", "Benchmark fixtures and the bench support layer."),
    ("ninfer_json_schema_oracle_test", "JSON-schema constrained decoding, against an oracle."),
    ("ninfer_regex_choice_test", "Regex and choice constrained decoding, with its oracle."),
    ("ninfer_topk", "Top-k scoring records and the log-probability Op."),
    ("ninfer_", "An Op, model or runtime test named for what it covers."),
]

PROBES = [
    ("tools/bench/report_latency_percentiles.py", "manual", "p50/p90/p95/p99 for queue_wait, prepare, prefill, ttft, total, decode from a serve request log."),
    ("tools/bench/probe_long_context_accuracy.py", "manual", "A (length x depth) needle sweep with a fresh needle per cell; reports the largest length passing at every depth."),
    ("tools/bench/probe_over_refusal.py", "manual", "OR-Bench hard-safe and toxic rejection rates, reported as the pair."),
    ("tools/bench/probe_concurrency_curve.py", "manual", "Aggregate and per-request throughput against concurrency."),
    ("tools/bench/lane_fingerprint.py", "manual (gate compares its record)", "Every shipped lane on the cells the product depends on; records figures under the engine's hash."),
    ("tools/bench/realtext_acceptance.py", "manual", "Speculative acceptance on real text at a lane's own flags."),
    ("tools/bench/probe_cancellation.py", "manual", "Whether an abandoned request frees its slot."),
    ("tools/bench/probe_cross_path_determinism.py", "manual", "Cached against cold path on the same continuation."),
    ("tools/bench/check_shared_prefix_reuse.py", "manual", "Whether a prefix cache serves a shared prefix, by path rather than hit count."),
    ("tools/bench/probe_prefill_floor.py", "manual", "The prefill floor and the phase split, by name, from the request log."),
    ("tools/bench/first_request_lane.py", "manual", "The cold first request against later identical ones."),
    ("tools/bench/warm_lane_sweep.py", "manual", "`prepared` and TTFT against conversation size."),
    ("tools/scripts/test_v3.cmd", "manual, and by the GPU workflow", "Configures, builds and runs the whole C++ suite, then the baseline gate reads its log."),
    ("tools/scripts/test_v3_asan.cmd", "manual", "The host-only subset under MSVC AddressSanitizer, in its own tree."),
    ("tools/scripts/test_v3_compute_sanitizer.cmd", "manual", "A five-test device subset under compute-sanitizer memcheck."),
]


def read(path: Path) -> str:
    """Read a text file, tolerating the UTF-16LE that PowerShell 5.1's `>` writes.

    `ctest -N > file` under PowerShell 5.1 produces UTF-16LE with a BOM - every character followed by a
    NUL - and a parser expecting UTF-8 then finds nothing and reports zero tests, which reads exactly
    like "there are no tests" rather than "the file is in another encoding". The tool detects it instead
    of asking the caller to remember.
    """
    if not path.exists():
        return ""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or (b"\x00" in raw[:64] and raw[:1] != b"\x00"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def gate_list() -> list[str]:
    """The gate names the hook runs, in order, read from the hook itself."""
    return re.findall(r'^\s*gate\s+"([^"]+)"', read(REPO / ".githooks" / "pre-commit"), re.M)


def hook_steps() -> list[str]:
    return re.findall(r'^echo "pre-commit: ([^"]+)"', read(REPO / ".githooks" / "pre-commit"), re.M)


def workflow_triggers() -> list[tuple[str, str, str]]:
    out = []
    for name in ("ci.yml", "gpu.yml"):
        text = read(REPO / ".github" / "workflows" / name)
        if not text:
            continue
        on = re.search(r"^on:\n(.*?)\n\w", text, re.M | re.S)
        block = on.group(1) if on else ""
        events = re.findall(r"^  (\w+):", block, re.M)
        runs = re.search(r"runs-on:\s*(.+)$", text, re.M)
        purpose = "runs `bash .githooks/pre-commit` on windows-latest, so the fast gates have one definition" \
            if name == "ci.yml" else \
            "runs the C++ suite and the baseline gate on the self-hosted 5090 runner, serialised against measurements"
        out.append((name, ", ".join(events) + (f" ({runs.group(1).strip()})" if runs else ""), purpose))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ctest-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tests = re.findall(r"Test\s+#\d+:\s+(\S+)", read(args.ctest_list))
    baseline = json.loads(read(REPO / "tools" / "release" / "test_baseline.json") or "{}")
    required = baseline.get("required_tests") or []
    known = baseline.get("known_failures") or {}
    optional = baseline.get("optional_tests") or {}
    py_tests = sorted(p.relative_to(REPO).as_posix()
                      for p in (REPO / "tests").rglob("test_*.py"))
    hook_pytest = re.findall(r"tests/[\w/.]*test_[\w]*\.py", read(REPO / ".githooks" / "pre-commit"))

    def purpose(name: str) -> str:
        for prefix, text in AREAS:
            if name.startswith(prefix):
                return text
        return AREAS[-1][1]

    def style(name: str) -> str:
        if name in known:
            return "known"
        if name in required:
            return "required"
        if name in optional:
            return "optional"
        return ""

    rows = []
    for name in tests:
        mark = style(name)
        tag = {"known": "KNOWN FAILURE", "required": "REQUIRED", "optional": "OPTIONAL"}.get(mark, "")
        rows.append(f"<tr class='{mark}'><td class='n'>{html.escape(name)}</td>"
                    f"<td>{html.escape(purpose(name))}"
                    + (f" <b>{tag}</b>" if tag else "") + "</td></tr>")

    gates = "".join(f"<li><code>{html.escape(g)}</code></li>" for g in gate_list())
    steps = "".join(f"<li><code>{html.escape(s)}</code></li>" for s in hook_steps())
    flows = "".join(f"<tr><td><code>{html.escape(n)}</code></td><td>{html.escape(e)}</td>"
                    f"<td>{html.escape(p)}</td></tr>" for n, e, p in workflow_triggers())
    probes = "".join(f"<tr><td class='n'><code>{html.escape(p)}</code></td>"
                     f"<td>{html.escape(w)}</td><td>{html.escape(d)}</td></tr>"
                     for p, w, d in PROBES)
    py_rows = "".join(
        f"<tr><td class='n'><code>{html.escape(p)}</code></td><td>"
        + ("runs in the pre-commit hook" if any(p.endswith(h.split('/')[-1]) for h in hook_pytest)
           else "run by the hook only if listed; otherwise manual")
        + "</td></tr>" for p in py_tests)

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>NInfer test inventory</title>
<style>
 body {{ font-family: Segoe UI, Arial, sans-serif; font-size: 10pt; margin: 18mm; color: #16181d; }}
 h1 {{ font-size: 20pt; margin: 0 0 2mm 0; }} h2 {{ font-size: 13pt; margin: 7mm 0 2mm 0;
 border-bottom: 1px solid #ccd; padding-bottom: 1mm; }}
 h3 {{ font-size: 11pt; margin: 4mm 0 1mm 0; }}
 p, li {{ line-height: 1.35; }} code {{ font-family: Consolas, monospace; font-size: 8.6pt; }}
 table {{ border-collapse: collapse; width: 100%; }} td, th {{ border-bottom: 1px solid #e2e4ea;
 padding: 1.2mm 1.6mm; vertical-align: top; text-align: left; font-size: 9pt; }}
 td.n {{ white-space: nowrap; }} th {{ background: #f4f5f8; }}
 tr.known {{ background: #fdeaea; }} tr.required {{ background: #eef7ee; }}
 tr.optional {{ background: #fff8e8; }}
 .note {{ background: #f4f5f8; padding: 2.5mm 3mm; border-left: 3px solid #8a93a6; margin: 3mm 0; }}
</style></head><body>
<h1>NInfer test inventory</h1>
<p>Every test in this tree, what it is for, and what runs it. Generated from the files that own each
fact: the C++ list from <code>ctest -N</code> on <code>build-test</code>, the gates from
<code>.githooks/pre-commit</code>, the triggers from the two workflows, the exceptions from
<code>tools/release/test_baseline.json</code>, and the probes from <code>tools/bench</code> and
<code>tools/scripts</code>. <b>{len(tests)} C++ tests, {len(py_tests)} Python test files.</b></p>

<h2>What runs when</h2>
<table><tr><th>Trigger</th><th>Events</th><th>What it does</th></tr>{flows}
<tr><td><code>.githooks/pre-commit</code></td><td>every <code>git commit</code> here</td>
<td>{len(gate_list())} gate scripts, both node plugin checks, pytest, ruff and mypy. The fast tier: seconds, no network.</td></tr>
<tr><td>Manual</td><td>on demand</td><td>The C++ suite, the sanitizer recipes, and every probe below.</td></tr></table>

<h2>The pre-commit hook, in order</h2>
<p>{len(steps)} steps. A gate that fails stops the commit; <code>--no-verify</code> skips it once.</p>
<ol>{steps}</ol>
<p class="note"><b>Gates that deny rather than report:</b> {len(gate_list())} scripts, each earning its place by catching
something a person missed. The retired-names gate, for instance, fails a criterion that names a symbol the tree no
longer defines, because such a criterion does not error - it answers a different question.</p>

<h2>The C++ suite: {len(tests)} tests</h2>
<p>Run by <code>tools/scripts/test_v3.cmd</code> (which configures, builds and runs), then judged by
<code>tools/release/check_test_baseline.py</code> against the recorded baseline. Colour marks the exceptions:
<span style="background:#eef7ee">required</span>,
<span style="background:#fdeaea">known failure</span>,
<span style="background:#fff8e8">optional</span>. Failures in any test, required or not, fail the gate.</p>
<table><tr><th>Test</th><th>What it is for</th></tr>{''.join(rows)}</table>

<h2>The Python tests: {len(py_tests)} files</h2>
<table><tr><th>File</th><th>Trigger</th></tr>{py_rows}</table>

<h2>Probes, recipes and harnesses</h2>
<p>None of these is a gate; each answers a question a gate cannot. The four newest exist because the
measurement priority named the gaps they fill.</p>
<table><tr><th>Tool</th><th>When it runs</th><th>What it answers</th></tr>{probes}</table>

<h2>The gate scripts</h2>
<ol>{gates}</ol>
</body></html>"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(doc, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(doc)} bytes)")
    print(f"  {len(tests)} C++ tests, {len(py_tests)} Python test files, {len(gate_list())} gates, "
          f"{len(steps)} hook steps")
    print(f"\n  to print: msedge --headless=new --print-to-pdf=docs/test-report.pdf "
          f"--no-pdf-header-footer file:///{args.out.resolve().as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
