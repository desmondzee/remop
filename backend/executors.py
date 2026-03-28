"""Dedicated thread pools so vision/WebSocket work cannot starve behind Gemini + disk I/O."""

from __future__ import annotations

import concurrent.futures
import os

# Vision: decode, YOLO infer, postprocess — must stay responsive while agent runs.
_vision_workers = max(2, int(os.environ.get("VISION_EXECUTOR_WORKERS", "8")))
VISION = concurrent.futures.ThreadPoolExecutor(
    max_workers=_vision_workers,
    thread_name_prefix="vision",
)

# Agent: sync Gemini client + parsing — separate pool so it never occupies all default workers.
_agent_workers = max(2, int(os.environ.get("AGENT_EXECUTOR_WORKERS", "4")))
AGENT = concurrent.futures.ThreadPoolExecutor(
    max_workers=_agent_workers,
    thread_name_prefix="agent",
)


def shutdown_pools() -> None:
    VISION.shutdown(wait=True, cancel_futures=False)
    AGENT.shutdown(wait=True, cancel_futures=False)
