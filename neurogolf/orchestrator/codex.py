"""Construct and run one isolated noninteractive Codex epoch."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from neurogolf.config.models import CodexConfig
from neurogolf.judge.reports import utc_now

DEFAULT_CODEX_PROFILE = "neurogolf-xhigh"


@dataclass(frozen=True)
class CodexInvocation:
    task_num: int
    workspace: Path
    prompt_path: Path
    event_log: Path
    final_message: Path
    output_schema: Path
    mode: Literal["fresh", "resume"] = "fresh"
    session_id: str | None = None
    profile: str | None = DEFAULT_CODEX_PROFILE


@dataclass
class CodexEpochResult:
    task_num: int
    status: str
    return_code: int | None
    timed_out: bool
    started_at: str
    completed_at: str
    event_log: str
    final_message: str
    session_id: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_codex_command(invocation: CodexInvocation, config: CodexConfig) -> list[str]:
    """Build the verified codex-cli 0.144 noninteractive command shape."""

    exec_options = [config.executable, "exec"]
    if invocation.profile:
        exec_options.extend(["--profile", invocation.profile])
    else:
        exec_options.extend(
            [
                "--model",
                config.model,
                "--sandbox",
                "workspace-write",
                "--config",
                'approval_policy="never"',
                "--config",
                f'model_reasoning_effort="{config.reasoning_effort}"',
            ]
        )
    exec_options.extend(["--strict-config", "--cd", str(invocation.workspace)])
    output_options = [
        "--json",
        "--output-schema",
        str(invocation.output_schema),
        "--output-last-message",
        str(invocation.final_message),
    ]
    prompt = invocation.prompt_path.read_text(encoding="utf-8")
    if invocation.mode == "resume":
        if not invocation.session_id:
            raise ValueError("Resume invocation requires a session ID")
        return [*exec_options, "resume", *output_options, invocation.session_id, prompt]
    return [*exec_options, *output_options, prompt]


def _extract_session_id(event_log: Path) -> str | None:
    if not event_log.is_file():
        return None
    for line in event_log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in ("thread_id", "session_id"):
            if isinstance(event.get(key), str):
                return event[key]
        if isinstance(event.get("thread"), dict) and isinstance(event["thread"].get("id"), str):
            return event["thread"]["id"]
    return None


async def run_codex_epoch(invocation: CodexInvocation, config: CodexConfig) -> CodexEpochResult:
    invocation.event_log.parent.mkdir(parents=True, exist_ok=True)
    invocation.final_message.parent.mkdir(parents=True, exist_ok=True)
    command = build_codex_command(invocation, config)
    started = utc_now()
    timed_out = False
    return_code: int | None = None
    error: str | None = None
    with invocation.event_log.open("wb") as events:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=invocation.workspace,
            stdout=events,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            return_code = await asyncio.wait_for(process.wait(), timeout=config.timeout_seconds)
        except TimeoutError:
            timed_out = True
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await process.wait()
            return_code = process.returncode
            error = f"Codex epoch exceeded {config.timeout_seconds}s timeout"
        except asyncio.CancelledError:
            process.terminate()
            await process.wait()
            raise
    return CodexEpochResult(
        task_num=invocation.task_num,
        status="completed" if return_code == 0 else ("timeout" if timed_out else "failed"),
        return_code=return_code,
        timed_out=timed_out,
        started_at=started,
        completed_at=utc_now(),
        event_log=str(invocation.event_log),
        final_message=(
            invocation.final_message.read_text(encoding="utf-8", errors="replace")
            if invocation.final_message.is_file()
            else ""
        ),
        session_id=_extract_session_id(invocation.event_log),
        error=error,
    )
