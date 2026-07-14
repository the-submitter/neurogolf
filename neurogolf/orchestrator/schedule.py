"""Bounded parallel task scheduling with per-task failure isolation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any


async def run_bounded(
    items: Iterable[int],
    worker: Callable[[int], Awaitable[Any]],
    *,
    limit: int,
) -> dict[int, Any]:
    semaphore = asyncio.Semaphore(limit)
    results: dict[int, Any] = {}

    async def one(item: int) -> None:
        async with semaphore:
            try:
                results[item] = await worker(item)
            except Exception as error:
                results[item] = {"status": "orchestrator_error", "error": str(error)}

    tasks = [asyncio.create_task(one(item), name=f"neurogolf-task-{item:03d}") for item in items]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return results
