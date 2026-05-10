"""File-based run logging helpers."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any


LOGGER_ROOT = "job_application_optimizer"


def _clean(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    text = " ".join(text.replace("|", "/").split())
    return text or "-"


def setup_run_logger(run_name: str, log_dir: Path | str = "logs") -> tuple[logging.Logger, Path]:
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_root / f"{run_name}_{timestamp}.log"

    logger = logging.getLogger(f"{LOGGER_ROOT}.{run_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.info("log_created | path=%s", log_path)
    return logger, log_path


def log_event(logger: logging.Logger | None, event: str, **fields: Any) -> None:
    if logger is None:
        return
    payload = " | ".join([event, *(f"{key}={_clean(value)}" for key, value in fields.items())])
    logger.info(payload)


__all__ = ["log_event", "setup_run_logger"]
