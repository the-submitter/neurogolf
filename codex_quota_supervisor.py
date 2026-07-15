#!/usr/bin/env python3
"""Monitor Codex usage and redeem one reset credit when quota reaches zero.

This client uses the experimental JSONL-over-stdio ``codex app-server``
protocol exposed by the installed Codex CLI. Protocol methods are intentionally
kept in constants near the top of the file so future CLI changes are easy to
audit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time
from typing import Any
import uuid


READ_RATE_LIMITS_METHOD = "account/rateLimits/read"
CONSUME_RESET_METHOD = "account/rateLimitResetCredit/consume"


class AppServerError(RuntimeError):
    """Raised when the app-server process or protocol request fails."""


class AppServerClient:
    def __init__(
        self,
        *,
        codex_bin: str,
        cwd: Path,
        stderr_path: Path,
        timeout: float,
    ) -> None:
        self.codex_bin = codex_bin
        self.cwd = cwd
        self.stderr_path = stderr_path
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self._stderr_file: Any = None
        self._messages: queue.Queue[dict[str, Any] | BaseException | None] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._next_request_id = 1

    def start(self) -> None:
        if self.process is not None:
            raise AppServerError("app-server client is already started")
        self._stderr_file = self.stderr_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [self.codex_bin, "app-server", "--stdio"],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            name="codex-app-server-reader",
            daemon=True,
        )
        self._reader_thread.start()

        response = self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "neurogolf-quota-supervisor",
                    "version": "1.0.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        if "userAgent" not in response:
            raise AppServerError("unexpected initialize response from app-server")
        self.notify("initialized")

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for raw_line in self.process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as error:
                    self._messages.put(AppServerError(f"non-JSON app-server output: {line!r}"))
                    self._messages.put(error)
                    continue
                if isinstance(message, dict):
                    self._messages.put(message)
        except BaseException as error:
            self._messages.put(error)
        finally:
            self._messages.put(None)

    def _send(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise AppServerError("app-server is not running")
        if self.process.poll() is not None:
            raise AppServerError(f"app-server exited with code {self.process.returncode}")
        try:
            self.process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise AppServerError(f"failed to write to app-server: {error}") from error

    def notify(self, method: str, params: Any = None) -> None:
        message: dict[str, Any] = {"method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def request(self, method: str, params: Any = None) -> dict[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        message: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AppServerError(f"timed out waiting for {method!r}")
            try:
                received = self._messages.get(timeout=remaining)
            except queue.Empty as error:
                raise AppServerError(f"timed out waiting for {method!r}") from error
            if received is None:
                returncode = self.process.poll() if self.process is not None else None
                raise AppServerError(f"app-server stdout closed (exit code {returncode})")
            if isinstance(received, BaseException):
                raise AppServerError(str(received)) from received
            if received.get("id") != request_id:
                # Server notifications (including rate-limit updates) can arrive
                # between request and response. The next explicit read is the
                # authoritative full snapshot, so no merge is required here.
                continue
            if "error" in received:
                raise AppServerError(f"{method!r} failed: {received['error']}")
            result = received.get("result")
            if not isinstance(result, dict):
                raise AppServerError(f"{method!r} returned an invalid result: {result!r}")
            return result

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        try:
                            if os.name == "posix":
                                os.killpg(process.pid, signal.SIGKILL)
                            else:
                                process.kill()
                        except ProcessLookupError:
                            pass
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    def __enter__(self) -> AppServerClient:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def log(message: str) -> None:
    timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def load_state(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return {"resets_consumed": 0}
    except (OSError, json.JSONDecodeError) as error:
        raise AppServerError(f"cannot load supervisor state {path}: {error}") from error
    if not isinstance(value, dict):
        raise AppServerError(f"supervisor state must be a JSON object: {path}")
    value.setdefault("resets_consumed", 0)
    return value


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select_rate_limit(result: dict[str, Any], limit_id: str) -> dict[str, Any] | None:
    by_id = result.get("rateLimitsByLimitId")
    if isinstance(by_id, dict):
        selected = by_id.get(limit_id)
        if isinstance(selected, dict):
            return selected
    legacy = result.get("rateLimits")
    if isinstance(legacy, dict) and legacy.get("limitId") in (None, limit_id):
        return legacy
    return None


def quota_status(snapshot: dict[str, Any]) -> tuple[bool, int | None, str]:
    windows: list[tuple[str, int]] = []
    for name in ("primary", "secondary"):
        value = snapshot.get(name)
        if isinstance(value, dict) and isinstance(value.get("usedPercent"), int):
            windows.append((name, value["usedPercent"]))
    reached_type = snapshot.get("rateLimitReachedType")
    exhausted = any(used >= 100 for _, used in windows) or reached_type == "rate_limit_reached"
    remaining = min((max(0, 100 - used) for _, used in windows), default=None)
    window_text = ", ".join(f"{name}={used}% used" for name, used in windows)
    if reached_type:
        window_text = f"{window_text}, reached={reached_type}" if window_text else f"reached={reached_type}"
    return exhausted, remaining, window_text or "no rate-limit windows returned"


def window_fingerprint(limit_id: str, snapshot: dict[str, Any]) -> str:
    identity: dict[str, Any] = {"limit_id": limit_id}
    for name in ("primary", "secondary"):
        window = snapshot.get(name)
        if isinstance(window, dict):
            identity[name] = {
                "resetsAt": window.get("resetsAt"),
                "windowDurationMins": window.get("windowDurationMins"),
            }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def available_credits(result: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    summary = result.get("rateLimitResetCredits")
    if not isinstance(summary, dict):
        return 0, []
    count = summary.get("availableCount")
    credits = summary.get("credits")
    usable = []
    if isinstance(credits, list):
        usable = [
            credit
            for credit in credits
            if isinstance(credit, dict)
            and credit.get("status") == "available"
            and credit.get("resetType") in (None, "codexRateLimits")
        ]
    return count if isinstance(count, int) else len(usable), usable


def choose_credit(credits: list[dict[str, Any]]) -> str | None:
    if not credits:
        return None
    far_future = 2**63 - 1
    selected = min(
        credits,
        key=lambda credit: (
            credit.get("expiresAt") if isinstance(credit.get("expiresAt"), int) else far_future,
            credit.get("grantedAt") if isinstance(credit.get("grantedAt"), int) else far_future,
        ),
    )
    credit_id = selected.get("id")
    return credit_id if isinstance(credit_id, str) else None


def maybe_reset(
    *,
    client: AppServerClient,
    result: dict[str, Any],
    limit_id: str,
    state: dict[str, Any],
    state_path: Path,
    dry_run: bool,
    max_resets: int,
) -> bool:
    snapshot = select_rate_limit(result, limit_id)
    if snapshot is None:
        log(f"limit {limit_id!r} was not present; no reset attempted")
        return False

    exhausted, remaining, details = quota_status(snapshot)
    count, credits = available_credits(result)
    remaining_text = "unknown" if remaining is None else f"{remaining}%"
    log(f"limit={limit_id} remaining={remaining_text}; {details}; reset credits={count}")
    fingerprint = window_fingerprint(limit_id, snapshot)

    if not exhausted:
        changed = False
        pending = state.get("pending_reset")
        if isinstance(pending, dict):
            before = pending.get("available_count_before")
            if isinstance(before, int) and count < before:
                state["resets_consumed"] = int(state.get("resets_consumed", 0)) + 1
            state.pop("pending_reset", None)
            changed = True
        if state.pop("awaiting_recovery_fingerprint", None) is not None:
            changed = True
        if changed:
            save_state(state_path, state)
        return False

    retry_after = state.get("retry_after_epoch")
    if isinstance(retry_after, (int, float)) and time.time() < retry_after:
        log(f"reset endpoint cooldown active for {max(0, int(retry_after - time.time()))}s")
        return False

    if state.get("awaiting_recovery_fingerprint") == fingerprint:
        log("a reset already succeeded for this exhaustion; waiting to observe quota recovery")
        return False

    consumed = int(state.get("resets_consumed", 0))
    if max_resets > 0 and consumed >= max_resets:
        log(f"configured --max-resets={max_resets} reached; no reset attempted")
        return False
    if count <= 0:
        log("quota is exhausted, but no reset credit is available")
        return False
    if dry_run:
        log("DRY RUN: quota is exhausted; would consume exactly one reset credit")
        return False

    pending = state.get("pending_reset")
    if (
        not isinstance(pending, dict)
        or pending.get("fingerprint") != fingerprint
        or not isinstance(pending.get("idempotency_key"), str)
    ):
        pending = {
            "fingerprint": fingerprint,
            "idempotency_key": str(uuid.uuid4()),
            "credit_id": choose_credit(credits),
            "available_count_before": count,
            "created_at": dt.datetime.now().astimezone().isoformat(),
        }
        state["pending_reset"] = pending
        save_state(state_path, state)

    params = {"idempotencyKey": pending["idempotency_key"]}
    if pending.get("credit_id"):
        params["creditId"] = pending["credit_id"]
    log("quota is exhausted; consuming one reset credit")
    response = client.request(CONSUME_RESET_METHOD, params)
    outcome = response.get("outcome")
    log(f"reset outcome: {outcome}")

    state.pop("pending_reset", None)
    state["last_outcome"] = outcome
    state["last_attempt_at"] = dt.datetime.now().astimezone().isoformat()
    if outcome in {"reset", "alreadyRedeemed"}:
        state["resets_consumed"] = consumed + 1
        state["awaiting_recovery_fingerprint"] = fingerprint
        state.pop("retry_after_epoch", None)
        save_state(state_path, state)
        return True
    if outcome == "nothingToReset":
        state["retry_after_epoch"] = time.time() + 60
    elif outcome == "noCredit":
        state["retry_after_epoch"] = time.time() + 300
    else:
        state["retry_after_epoch"] = time.time() + 60
    save_state(state_path, state)
    return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor Codex quota and consume one reset credit at zero remaining usage."
    )
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--restart-delay", type=float, default=15.0)
    parser.add_argument("--post-reset-delay", type=float, default=2.0)
    parser.add_argument("--limit-id", default="codex")
    parser.add_argument(
        "--max-resets",
        type=int,
        default=0,
        help="maximum successful resets recorded in the state file; 0 uses all available credits",
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--state-file", default="logs/quota-supervisor-state.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and report quota but never call the reset method",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="perform one poll (and, unless --dry-run, one eligible reset) then exit",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    for name in ("poll_interval", "request_timeout", "restart_delay", "post_reset_delay"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.max_resets < 0:
        parser.error("--max-resets cannot be negative")
    if shutil.which(args.codex_bin) is None:
        parser.error(f"Codex executable not found: {args.codex_bin!r}")

    repo_root = Path(__file__).resolve().parent
    state_path = Path(args.state_file)
    if not state_path.is_absolute():
        state_path = repo_root / state_path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = state_path.parent / "quota-supervisor-app-server.stderr"
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")

    stop_event = threading.Event()

    def stop(signum: int, _frame: object) -> None:
        log(f"received signal {signum}; stopping")
        stop_event.set()

    for caught_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(caught_signal, stop)

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"another quota supervisor owns {lock_path}")
            return 2

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} started={dt.datetime.now().astimezone().isoformat()}\n")
        lock_file.flush()

        try:
            state = load_state(state_path)
        except AppServerError as error:
            log(str(error))
            return 2

        client: AppServerClient | None = None
        exit_code = 0
        try:
            while not stop_event.is_set():
                try:
                    if client is None:
                        log("starting persistent codex app-server")
                        client = AppServerClient(
                            codex_bin=args.codex_bin,
                            cwd=repo_root,
                            stderr_path=stderr_path,
                            timeout=args.request_timeout,
                        )
                        client.start()
                        log("app-server initialized")

                    result = client.request(READ_RATE_LIMITS_METHOD)
                    reset_succeeded = maybe_reset(
                        client=client,
                        result=result,
                        limit_id=args.limit_id,
                        state=state,
                        state_path=state_path,
                        dry_run=args.dry_run,
                        max_resets=args.max_resets,
                    )
                    if args.once:
                        break
                    delay = args.post_reset_delay if reset_succeeded else args.poll_interval
                    stop_event.wait(delay)
                except (AppServerError, OSError, KeyError, TypeError, ValueError) as error:
                    log(f"supervisor error: {error}")
                    if client is not None:
                        client.close()
                        client = None
                    if args.once:
                        exit_code = 1
                        break
                    stop_event.wait(args.restart_delay)
        finally:
            if client is not None:
                client.close()
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
