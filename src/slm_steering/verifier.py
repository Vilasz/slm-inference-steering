from __future__ import annotations

import re
import os
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from slm_steering.datasets import CodingProblem


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    verification_seconds: float
    error_type: str | None
    stdout: str
    stderr: str
    candidate_source: str


def check_correctness(
    problem: CodingProblem,
    completion: str,
    timeout_seconds: float,
    temp_root: Path | None = None,
) -> VerificationResult:
    candidate_source = build_candidate_source(problem, completion)
    script = compose_verification_script(problem, candidate_source)
    temp_root = temp_root or _default_temp_root()
    temp_root.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    temp_dir = temp_root / f"humaneval_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
        script_path = temp_dir / "candidate_check.py"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return VerificationResult(
            passed=False,
            verification_seconds=elapsed,
            error_type="timeout",
            stdout=_clip(exc.stdout or ""),
            stderr=_clip(exc.stderr or ""),
            candidate_source=candidate_source,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.perf_counter() - start
    passed = completed.returncode == 0
    return VerificationResult(
        passed=passed,
        verification_seconds=elapsed,
        error_type=None if passed else "runtime_error",
        stdout=_clip(completed.stdout),
        stderr=_clip(completed.stderr),
        candidate_source=candidate_source,
    )


def build_candidate_source(problem: CodingProblem, completion: str) -> str:
    code = normalize_completion(completion)
    if _defines_entry_point(code, problem.entry_point):
        prompt_prefix = _prompt_prefix_before_entry_point(problem.prompt, problem.entry_point)
        if prompt_prefix:
            return prompt_prefix.rstrip() + "\n\n" + code.rstrip() + "\n"
        return code.rstrip() + "\n"
    return problem.prompt.rstrip() + "\n" + code.rstrip() + "\n"


def compose_verification_script(problem: CodingProblem, candidate_source: str) -> str:
    return (
        "import faulthandler\n"
        "faulthandler.enable()\n\n"
        f"{candidate_source.rstrip()}\n\n"
        f"{problem.test.rstrip()}\n\n"
        f"check({problem.entry_point})\n"
    )


def normalize_completion(text: str) -> str:
    text = _extract_first_code_fence(text).strip("\n")
    text = _strip_common_preamble(text)
    text = _trim_after_obvious_non_solution(text)
    return text.rstrip()


def _extract_first_code_fence(text: str) -> str:
    match = re.search(
        r"```(?:python|py)?[^\S\r\n]*(?:\r?\n)?(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1)
    return text


def _strip_common_preamble(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if (
            stripped.startswith("def ")
            or stripped.startswith("from ")
            or stripped.startswith("import ")
            or stripped.startswith("return ")
            or stripped.startswith("if ")
            or stripped.startswith("for ")
            or stripped.startswith("while ")
            or stripped.startswith("try:")
            or stripped.startswith("#")
            or line.startswith((" ", "\t"))
        ):
            return "\n".join(lines[index:])
    return text


def _trim_after_obvious_non_solution(text: str) -> str:
    stop_patterns = [
        r"\n\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:",
        r"\n\s*def\s+check\s*\(",
        r"\n\s*#\s*Tests?\b",
        r"\n\s*#\s*Example\b",
        r"\n\s*print\s*\(",
    ]
    end = len(text)
    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            end = min(end, match.start())
    return text[:end]


def _defines_entry_point(code: str, entry_point: str) -> bool:
    pattern = rf"(^|\n)\s*def\s+{re.escape(entry_point)}\s*\("
    return re.search(pattern, code) is not None


def _prompt_prefix_before_entry_point(prompt: str, entry_point: str) -> str:
    pattern = rf"(^|\n)\s*def\s+{re.escape(entry_point)}\s*\("
    match = re.search(pattern, prompt)
    if not match:
        return ""
    return prompt[: match.start()].rstrip()


def _clip(text: str, limit: int = 4000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return textwrap.shorten(text, width=limit, placeholder="\n...[truncated]...")


def _default_temp_root() -> Path:
    configured = os.environ.get("SLM_STEERING_TMP")
    if configured:
        return Path(configured)
    return Path.cwd() / "runs" / "tmp"
