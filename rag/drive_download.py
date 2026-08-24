"""Download a publicly-shared PDF from Google Drive.

Django-free by design, like the rest of `rag/`: the management command in
`documents/` is the only Django-aware piece. That keeps this importable from a
plain script or a test without a settings module.

Only handles files shared as **"Anyone with the link"**. There is no OAuth here
and no service account — for a private file Drive returns its sign-in page,
which we detect and report rather than saving as a corrupt PDF.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import unquote

import requests

# Drive's own limit for a single anonymous download; also a sanity bound so a
# surprise HTML error page can't stream forever into the data directory.
MAX_BYTES = 200 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
TIMEOUT_SECONDS = 60

UC_ENDPOINT = "https://drive.google.com/uc"

# A Drive share link takes several shapes depending on where it was copied
# from. All of them carry the same 25-45 char file id.
_ID_PATTERNS = (
    re.compile(r"/file/d/([A-Za-z0-9_-]+)"),        # .../file/d/<id>/view
    re.compile(r"/d/([A-Za-z0-9_-]+)"),             # .../d/<id>
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),         # uc?id=<id>, open?id=<id>
)

# The large-file interstitial: Drive can't virus-scan big files, so instead of
# the bytes it returns an HTML page with a form that re-requests with a
# confirm token. Attribute order inside the tag is not stable, so the form tag
# and its hidden inputs are matched separately rather than in one pattern.
_FORM_RE = re.compile(r"<form[^>]*\bid=[\"']download-form[\"'][^>]*>", re.I)
_ACTION_RE = re.compile(r"\baction=[\"']([^\"']+)[\"']", re.I)
_HIDDEN_INPUT_RE = re.compile(r"<input[^>]*\btype=[\"']hidden[\"'][^>]*>", re.I)
_NAME_RE = re.compile(r"\bname=[\"']([^\"']+)[\"']", re.I)
_VALUE_RE = re.compile(r"\bvalue=[\"']([^\"']*)[\"']", re.I)

# Content-Disposition: attachment; filename="x.pdf"; filename*=UTF-8''x.pdf
_FILENAME_STAR_RE = re.compile(r"filename\*\s*=\s*[^']*'[^']*'([^;]+)", re.I)
_FILENAME_RE = re.compile(r"filename\s*=\s*\"([^\"]+)\"|filename\s*=\s*([^;]+)", re.I)

# Anything that has no business in a filename we control.
_UNSAFE_CHARS_RE = re.compile(r"[\x00-\x1f<>:\"/\\|?*]+")
MAX_FILENAME_LENGTH = 120


class DriveDownloadError(RuntimeError):
    """The file could not be fetched, or what came back was not a PDF."""


def extract_file_id(link: str) -> str:
    """Pull the Drive file id out of any of the usual share-link shapes.

    A bare file id is accepted too, so you can paste either.
    """
    link = (link or "").strip()
    if not link:
        raise DriveDownloadError("Empty Google Drive link.")

    # A bare id: no scheme, no slashes.
    if "/" not in link and re.fullmatch(r"[A-Za-z0-9_-]{10,}", link):
        return link

    for pattern in _ID_PATTERNS:
        match = pattern.search(link)
        if match:
            return match.group(1)

    raise DriveDownloadError(
        f"Could not find a file id in {link!r}. Expected a link like "
        "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing"
    )


def safe_filename(raw: str, *, fallback: str) -> str:
    """Reduce a remote-supplied name to something safe to write in `data/`.

    The name comes from a header on a public URL, so it is untrusted input:
    it could carry path separators, traversal segments, or control characters.
    """
    name = Path((raw or "").strip().replace("\\", "/")).name
    name = _UNSAFE_CHARS_RE.sub("_", name).strip(" .")

    if not name or name in {".", ".."}:
        name = fallback

    stem, dot, extension = name.rpartition(".")
    if not dot or extension.lower() != "pdf":
        stem, extension = name, "pdf"

    stem = stem[:MAX_FILENAME_LENGTH] or fallback
    return f"{stem}.{extension.lower()}"


def filename_from_response(response: requests.Response, *, fallback: str) -> str:
    """Work out what to call the file, preferring Drive's own name for it."""
    disposition = response.headers.get("Content-Disposition", "")

    # RFC 5987 form wins: it is the one that carries non-ASCII names correctly.
    star = _FILENAME_STAR_RE.search(disposition)
    if star:
        return safe_filename(unquote(star.group(1).strip().strip('"')), fallback=fallback)

    plain = _FILENAME_RE.search(disposition)
    if plain:
        return safe_filename(
            (plain.group(1) or plain.group(2) or "").strip(), fallback=fallback
        )

    return safe_filename("", fallback=fallback)


def _confirm_params(page_html: str) -> dict[str, str] | None:
    """Rebuild the interstitial's form submission as query parameters.

    Returns None when the page isn't the confirm interstitial.
    """
    form = _FORM_RE.search(page_html)
    if not form:
        return None

    action_match = _ACTION_RE.search(form.group(0))
    if not action_match:
        return None

    params: dict[str, str] = {}
    for field in _HIDDEN_INPUT_RE.finditer(page_html):
        name = _NAME_RE.search(field.group(0))
        value = _VALUE_RE.search(field.group(0))
        if name:
            params[name.group(1)] = html.unescape(value.group(1) if value else "")

    if "confirm" not in params:
        return None

    params["__action__"] = html.unescape(action_match.group(1))
    return params


def _looks_like_html(response: requests.Response) -> bool:
    return "text/html" in response.headers.get("Content-Type", "").lower()


def _diagnose_html(page_html: str) -> str:
    """Turn Drive's HTML response into a message worth reading."""
    lowered = page_html.lower()
    if "accounts.google.com" in lowered or "sign in" in lowered:
        return (
            "Drive returned a sign-in page. The file is not public — set its "
            'sharing to "Anyone with the link" (Viewer).'
        )
    if "quota" in lowered or "too many" in lowered:
        return (
            "Drive refused the download: this file has exceeded its public "
            "download quota. Try again later, or host a copy elsewhere."
        )
    if "no longer exists" in lowered or "not found" in lowered:
        return "Drive says the file does not exist. Check the link and file id."
    return (
        "Drive returned an HTML page instead of a file. The link is probably "
        "not a direct-downloadable public file."
    )


def download_drive_pdf(
    link: str,
    dest_dir: Path,
    *,
    name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download the shared PDF at `link` into `dest_dir`.

    The saved filename comes from Drive's own `Content-Disposition` header
    unless `name` overrides it, so `fetch_drive <url>` needs nothing but the
    URL. Returns the path written.

    Raises `DriveDownloadError` with an actionable message on any failure —
    including a successful HTTP response whose body turns out not to be a PDF.
    """
    file_id = extract_file_id(link)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        response = session.get(
            UC_ENDPOINT,
            params={"export": "download", "id": file_id},
            stream=True,
            timeout=TIMEOUT_SECONDS,
        )
        response = _resolve_confirmation(session, response, file_id)

        if _looks_like_html(response):
            raise DriveDownloadError(_diagnose_html(response.text[:8000]))

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise DriveDownloadError(f"Drive returned HTTP {response.status_code}.") from exc

        filename = (
            safe_filename(name, fallback=file_id)
            if name
            else filename_from_response(response, fallback=file_id)
        )
        destination = dest_dir / filename

        if destination.exists() and not overwrite:
            raise DriveDownloadError(
                f"{destination} already exists. Pass --overwrite to replace it, "
                "or --name to save it under a different name."
            )

        # Written to a sibling temp file and renamed, so an interrupted
        # download never leaves a half-written PDF where ingest_docs will
        # find it and try to parse it.
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            total = _stream_to_file(response, partial)
            _verify_pdf(partial, total)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)

    return destination


def _resolve_confirmation(
    session: requests.Session,
    response: requests.Response,
    file_id: str,
) -> requests.Response:
    """Follow the large-file confirm interstitial, if Drive served one."""
    if not _looks_like_html(response):
        return response

    # `response.text` consumes the stream, which is fine: an HTML body is
    # small, and we're about to re-request either way.
    page_html = response.text

    params = _confirm_params(page_html)
    if params:
        action = params.pop("__action__")
        return session.get(action, params=params, stream=True, timeout=TIMEOUT_SECONDS)

    # Older interstitial: the token arrives as a cookie rather than a form.
    for cookie_name, value in session.cookies.items():
        if cookie_name.startswith("download_warning"):
            return session.get(
                UC_ENDPOINT,
                params={"export": "download", "id": file_id, "confirm": value},
                stream=True,
                timeout=TIMEOUT_SECONDS,
            )

    # Not an interstitial — a real error page. Hand it back so the caller can
    # diagnose it with the full body.
    response._content = page_html.encode(response.encoding or "utf-8", "replace")
    return response


def _stream_to_file(response: requests.Response, path: Path) -> int:
    written = 0
    with path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if not chunk:
                continue
            written += len(chunk)
            if written > MAX_BYTES:
                raise DriveDownloadError(
                    f"Download exceeded {MAX_BYTES // (1024 * 1024)}MB and was "
                    "aborted. Is this really the right file?"
                )
            handle.write(chunk)
    return written


def _verify_pdf(path: Path, total: int) -> None:
    """Fail loudly unless the bytes on disk are actually a PDF.

    Drive answers many failures with HTTP 200 and an HTML body, so the status
    code alone proves nothing. The magic number does.
    """
    if total == 0:
        raise DriveDownloadError("Downloaded 0 bytes from Drive.")

    with path.open("rb") as handle:
        header = handle.read(5)

    if not header.startswith(b"%PDF"):
        snippet = header.decode("utf-8", "replace").strip()
        raise DriveDownloadError(
            f"Downloaded {total} bytes, but the file does not start with %PDF "
            f"(got {snippet!r}). The link may point to a Google Doc rather than "
            "an uploaded PDF — use File > Download > PDF, upload that, and share "
            "the upload."
        )
