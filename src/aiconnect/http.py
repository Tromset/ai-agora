"""Tiny stdlib-only HTTP helpers shared by providers (honours HTTPS_PROXY via urllib)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterator

from .errors import ProviderError

DEFAULT_TIMEOUT = 120.0


def _request(url: str, headers: dict[str, str], payload: dict[str, Any]) -> urllib.request.Request:
    data = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **headers}
    return urllib.request.Request(url, data=data, headers=all_headers, method="POST")


def _raise_http_error(url: str, err: urllib.error.HTTPError) -> None:
    body = err.read().decode("utf-8", "replace")[:500]
    raise ProviderError(f"HTTP {err.code} from {url}: {body}") from err


def post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """POST a JSON payload and return the decoded JSON response."""
    try:
        with urllib.request.urlopen(_request(url, headers, payload), timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        _raise_http_error(url, err)
    except urllib.error.URLError as err:
        raise ProviderError(f"Cannot reach {url}: {err.reason}") from err
    except json.JSONDecodeError as err:
        raise ProviderError(f"Invalid JSON from {url}: {err}") from err
    raise AssertionError("unreachable")


def post_lines(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = DEFAULT_TIMEOUT
) -> Iterator[str]:
    """POST a JSON payload and yield the response body line by line (for SSE / NDJSON streams)."""
    try:
        with urllib.request.urlopen(_request(url, headers, payload), timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line:
                    yield line
    except urllib.error.HTTPError as err:
        _raise_http_error(url, err)
    except urllib.error.URLError as err:
        raise ProviderError(f"Cannot reach {url}: {err.reason}") from err


def iter_sse_data(lines: Iterator[str]) -> Iterator[str]:
    """Extract the `data:` payloads from Server-Sent Events lines, stopping at `[DONE]`."""
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield data
