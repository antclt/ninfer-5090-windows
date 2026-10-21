"""Tests for Python 3.11+ resolution in launcher_env.bat and checkpoint completeness in
download_model.py / build_model.py (Issues #12 and #13).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "release"))

import build_model  # noqa: E402
import download_model  # noqa: E402


def _write_fake_python(directory: Path, name: str, major: int, minor: int) -> Path:
    """Create a batch shim that behaves like a Python interpreter of version (major, minor)."""
    shim = directory / f"{name}.cmd"
    exe_path = directory / f"{name}_real.exe"
    exe_path.write_bytes(b"")
    site_dir = directory / f"site_{name}"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "sitecustomize.py").write_text(
        f"import sys\n"
        f"sys.version_info = ({major}, {minor}, 10, 'final', 0)\n"
        f"sys.executable = r'{exe_path}'\n",
        encoding="utf-8",
    )
    shim.write_bytes(
        (
            "@echo off\r\n"
            f'set "PYTHONPATH={site_dir}"\r\n'
            f'"{sys.executable}" %*\r\n'
        ).encode("utf-8")
    )
    return exe_path


def _write_fake_py_launcher(directory: Path, versions: dict[str, tuple[int, int]]) -> dict[str, Path]:
    """Create a fake `py.cmd` supporting version flags like `-3.11` and `-3`."""
    exes: dict[str, Path] = {}
    for flag, (major, minor) in versions.items():
        tag = flag.lstrip("-").replace(".", "_")
        exes[flag] = _write_fake_python(directory, f"py_{tag}", major, minor)

    lines = ["@echo off"]
    for flag in versions:
        tag = flag.lstrip("-").replace(".", "_")
        lines.append(f'if "%~1"=="{flag}" goto :run_{tag}')
    lines.append("exit /b 1")
    for flag in versions:
        tag = flag.lstrip("-").replace(".", "_")
        shim_cmd = directory / f"py_{tag}.cmd"
        lines.append(f":run_{tag}")
        lines.append("shift")
        lines.append(f'call "{shim_cmd}" %1 %2 %3 %4 %5 %6 %7 %8 %9')
        lines.append("exit /b %ERRORLEVEL%")
    (directory / "py.cmd").write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return exes


def _run_launcher_env(tmp_path: Path, extra_env: dict[str, str]) -> str:
    """Run launcher_env.bat in a hermetic environment and return the resolved PYTHON_EXE."""
    # Isolate from C:\vllm-env and %USERPROFILE% uv paths by testing a copy of launcher_env.bat
    # with VLLM_PYTHON pointed into tmp_path unless overridden.
    env_bat = tmp_path / "launcher_env.bat"
    text = (REPO / "launcher_env.bat").read_text(encoding="utf-8")
    text = text.replace(r"C:\vllm-env\Scripts\python.exe", str(tmp_path / "no_vllm" / "python.exe"))
    env_bat.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))

    probe = tmp_path / "probe.bat"
    probe.write_bytes(
        (
            "@echo off\r\n"
            f'call "{env_bat}"\r\n'
            "echo RESOLVED=%PYTHON_EXE%\r\n"
        ).encode("utf-8")
    )
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    env = {
        "SystemRoot": sys_root,
        "WINDIR": sys_root,
        "COMSPEC": os.environ.get("COMSPEC", rf"{sys_root}\System32\cmd.exe"),
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "USERPROFILE": str(tmp_path / "empty_profile"),
        "PATH": f"{tmp_path};{sys_root}\\System32",
        **extra_env,
    }
    res = subprocess.run(
        ["cmd.exe", "/c", str(probe)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    for line in res.stdout.splitlines():
        if line.startswith("RESOLVED="):
            return line.split("=", 1)[1].strip()
    return ""


def _sys32_path(bin_dir: Path) -> str:
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    return f"{bin_dir};{sys_root}\\System32"


def test_launcher_env_accepts_python_312_on_path(tmp_path: Path) -> None:
    """Issue #12: Python 3.12 on PATH with no Python 3.11 installed must resolve PYTHON_EXE."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = _write_fake_python(bin_dir, "python", 3, 12)
    resolved = _run_launcher_env(tmp_path, {"PATH": _sys32_path(bin_dir)})
    assert resolved == str(exe)


def test_launcher_env_accepts_py_launcher_with_312_only(tmp_path: Path) -> None:
    """Issue #12: `py` launcher with 3.12 (via `py -3`) and no 3.11 must resolve PYTHON_EXE."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exes = _write_fake_py_launcher(bin_dir, {"-3": (3, 12)})
    resolved = _run_launcher_env(tmp_path, {"PATH": _sys32_path(bin_dir)})
    assert resolved == str(exes["-3"])


def test_launcher_env_rejects_python_310(tmp_path: Path) -> None:
    """Python < 3.11 must still be rejected."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_python(bin_dir, "python", 3, 10)
    _write_fake_py_launcher(bin_dir, {"-3": (3, 10)})
    resolved = _run_launcher_env(tmp_path, {"PATH": _sys32_path(bin_dir)})
    assert resolved == ""


def test_launcher_env_preserves_caller_python_exe(tmp_path: Path) -> None:
    """Caller-supplied PYTHON_EXE must not be clobbered by launcher_env.bat."""
    custom = r"C:\custom\venv\Scripts\python.exe"
    resolved = _run_launcher_env(tmp_path, {"PYTHON_EXE": custom})
    assert resolved == custom


def test_dir_stats_excludes_huggingface_cache_dir(tmp_path: Path) -> None:
    """Issue #13: `.cache/huggingface/download/*.incomplete` must not count as checkpoint files."""
    dest = tmp_path / "Qwen3.8-27B-NVFP4-QUASAR"
    cache_dl = dest / ".cache" / "huggingface" / "download"
    cache_dl.mkdir(parents=True)
    (cache_dl / "shard.1234abcd.incomplete").write_bytes(b"x" * 4096)
    (cache_dl / "shard.lock").write_bytes(b"")

    count, total = download_model.dir_stats(str(dest))
    assert count == 0
    assert total == 0


def _populate_complete_checkpoint(dest: Path, *, sharded: bool = True, draft_only: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
    if not draft_only:
        (dest / "generation_config.json").write_text("{}", encoding="utf-8")
        (dest / "tokenizer.json").write_text("{}", encoding="utf-8")
        (dest / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (dest / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (dest / "video_preprocessor_config.json").write_text("{}", encoding="utf-8")
        (dest / "hf_quant_config.json").write_text("{}", encoding="utf-8")
    if sharded:
        s1 = "model-00001-of-00002.safetensors"
        s2 = "model-00002-of-00002.safetensors"
        (dest / s1).write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
        (dest / s2).write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
        (dest / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": s1, "b": s2}}),
            encoding="utf-8",
        )
    else:
        (dest / "model.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")


def test_is_source_complete_detects_incomplete_and_missing_shards(tmp_path: Path) -> None:
    """Issue #13: partial checkpoint directories must be detected as incomplete."""
    repo = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
    local = download_model.SOURCES[repo][0]
    dest = tmp_path / local

    # 1. Directory exists with only .cache/*.incomplete
    cache_dl = dest / ".cache" / "huggingface" / "download"
    cache_dl.mkdir(parents=True)
    (cache_dl / "part.incomplete").write_bytes(b"partial")
    assert download_model.is_source_complete(str(tmp_path), repo) is False
    assert download_model.report(str(tmp_path), [repo]) is False

    # 2. Clean .incomplete, write config + index + 1 of 2 shards -> still incomplete
    (cache_dl / "part.incomplete").unlink()
    _populate_complete_checkpoint(dest, sharded=True)
    (dest / "model-00002-of-00002.safetensors").unlink()
    assert download_model.is_source_complete(str(tmp_path), repo) is False
    assert download_model.report(str(tmp_path), [repo]) is False

    # 3. Restore second shard -> complete
    (dest / "model-00002-of-00002.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    assert download_model.is_source_complete(str(tmp_path), repo) is True
    assert download_model.report(str(tmp_path), [repo]) is True

    # 4. If an .incomplete file is present alongside shards, still treated as incomplete
    (cache_dl / "retry.incomplete").write_bytes(b"partial")
    assert download_model.is_source_complete(str(tmp_path), repo) is False
    (cache_dl / "retry.incomplete").unlink()

    # 5. Stale revision in sources.json marks checkpoint incomplete
    download_model.record_revision(str(tmp_path), repo, local, "0" * 40)
    assert download_model.is_source_complete(str(tmp_path), repo) is False
    download_model.record_revision(str(tmp_path), repo, local, download_model.SOURCES[repo][2])
    assert download_model.is_source_complete(str(tmp_path), repo) is True


def test_unsloth_source_requires_index_and_mtp_shard(tmp_path: Path) -> None:
    """unsloth/Qwen3.8-27B-NVFP4 has model.safetensors + model_mtp.safetensors; missing index must fail."""
    repo = "unsloth/Qwen3.8-27B-NVFP4"
    dest = tmp_path / download_model.SOURCES[repo][0]
    _populate_complete_checkpoint(dest, sharded=False)
    assert download_model.is_source_complete(str(tmp_path), repo) is False

    (dest / "model_mtp.safetensors").write_bytes(b"\x02\x00\x00\x00\x00\x00\x00\x00{}")
    (dest / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model.safetensors", "m": "model_mtp.safetensors"}}),
        encoding="utf-8",
    )
    assert download_model.is_source_complete(str(tmp_path), repo) is True


def test_build_model_refetches_interrupted_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #13: build_model.py must re-fetch a source whose directory exists from an interrupted download."""
    hf_src = tmp_path / "hf-src"
    dest_models = tmp_path / "models"

    quasar_repo = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
    bf16_repo = "Qwen/Qwen3.8-27B"
    dflash_repo = "z-lab/Qwen3.8-27B-DFlash2"

    # QUASAR has an interrupted directory (only .cache/*.incomplete)
    quasar_dir = hf_src / download_model.SOURCES[quasar_repo][0]
    (quasar_dir / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (quasar_dir / ".cache" / "huggingface" / "download" / "shard.incomplete").write_bytes(b"1234")

    # BF16 and DFlash2 are complete
    _populate_complete_checkpoint(hf_src / download_model.SOURCES[bf16_repo][0], sharded=True)
    _populate_complete_checkpoint(
        hf_src / download_model.SOURCES[dflash_repo][0], sharded=False, draft_only=True
    )

    fetched: list[str] = []

    def fake_fetch(repo: str, root: str) -> bool:
        fetched.append(repo)
        download_model.clean_incomplete(os.path.join(root, download_model.SOURCES[repo][0]))
        _populate_complete_checkpoint(Path(root) / download_model.SOURCES[repo][0], sharded=True)
        return True

    class DummyCompleted:
        returncode = 0

    monkeypatch.setattr(build_model, "fetch", fake_fetch)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: DummyCompleted())
    with patch.object(
        sys,
        "argv",
        ["build_model.py", "--line", "quasar", "--hf-src", str(hf_src), "--dest", str(dest_models)],
    ):
        rc = build_model.main()

    assert rc == 0
    assert fetched == [quasar_repo], f"Expected {quasar_repo} to be re-fetched, got {fetched}"
