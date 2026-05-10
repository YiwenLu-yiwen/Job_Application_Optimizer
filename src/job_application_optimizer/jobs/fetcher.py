"""Job page fetching with optional browser-render fallback."""

import os
from typing import Literal

import requests
from bs4 import BeautifulSoup

from job_application_optimizer.config import _read_int_env


FetchMode = Literal["auto", "requests", "browser"]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

DYNAMIC_PAGE_MARKERS = (
    "enable javascript",
    "javascript is disabled",
    "javascript is required",
    "please enable javascript",
    "checking your browser",
    "please wait while",
    "loading...",
)


def _fetch_mode() -> FetchMode:
    mode = (os.getenv("JOB_FETCH_MODE", "browser") or "browser").strip().lower()
    if mode not in {"auto", "requests", "browser"}:
        return "browser"
    return mode  # type: ignore[return-value]


def _request_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


def _fetch_with_requests(url: str, timeout: int) -> str:
    response = requests.get(url, timeout=timeout, headers=_request_headers())
    response.raise_for_status()
    return response.text


def _visible_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _needs_browser_render(html: str) -> bool:
    min_text_chars = _read_int_env("JOB_FETCH_MIN_TEXT_CHARS", 800)
    text = _visible_text(html)
    lower_text = text.lower()
    if any(marker in lower_text for marker in DYNAMIC_PAGE_MARKERS):
        return True
    return len(text) < min_text_chars


def _fetch_with_browser(url: str, timeout: int) -> str:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Playwright browser fallback is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    timeout_ms = _read_int_env("JOB_FETCH_BROWSER_TIMEOUT_MS", timeout * 1000)
    wait_ms = _read_int_env("JOB_FETCH_BROWSER_WAIT_MS", 1500)
    headless = (os.getenv("JOB_FETCH_BROWSER_HEADLESS", "true") or "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            try:
                context = browser.new_context(user_agent=USER_AGENT)
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 10_000))
                except PlaywrightTimeoutError:
                    pass
                if wait_ms > 0:
                    page.wait_for_timeout(wait_ms)
                return page.content()
            finally:
                browser.close()
    except PlaywrightError as exc:
        raise RuntimeError(
            "Playwright browser fallback failed. If this is the first browser run, execute "
            "`python -m playwright install chromium` and retry."
        ) from exc


def fetch_job_page(url: str, timeout: int = 30) -> str:
    mode = _fetch_mode()
    request_html = ""
    request_error: Exception | None = None

    if mode in {"auto", "requests"}:
        try:
            request_html = _fetch_with_requests(url, timeout)
        except requests.RequestException as exc:
            request_error = exc
            if mode == "requests":
                raise
        else:
            if mode == "requests" or not _needs_browser_render(request_html):
                return request_html

    if mode in {"auto", "browser"}:
        try:
            return _fetch_with_browser(url, timeout)
        except Exception as exc:
            if mode == "auto" and request_html:
                print(f"Browser fallback failed; continuing with requests HTML. Error: {exc}", flush=True)
                return request_html
            if request_error is not None:
                raise RuntimeError(f"Requests fetch failed ({request_error}); browser fallback also failed ({exc}).") from exc
            raise

    return request_html


__all__ = ["fetch_job_page"]
