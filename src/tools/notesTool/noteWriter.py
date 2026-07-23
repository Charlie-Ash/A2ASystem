# Sanitizes note filenames and writes note content to disk. (data/notes)
import re
from datetime import datetime
from pathlib import Path

from tools.notesTool.config import NOTES_DIR

# Only these characters are allowed through from the LLM-provided file_name,
# since it becomes part of a filesystem path -- this also rules out path
# traversal via "../" or an absolute path sneaking in.
_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_BASE_LENGTH = 50


def _sanitize_base_name(file_name: str) -> str:

    base = _SAFE_CHARS_RE.sub("_", file_name.strip()).strip("_")
    base = base[:_MAX_BASE_LENGTH].strip("_")

    if not base:
        # No usable name was provided (or it sanitized down to nothing) -- fall
        # back to a timestamp so the note still gets saved under a unique name.
        base = datetime.now().strftime("note_%Y%m%d_%H%M%S")

    return base


def save_note(content: str, file_name: str) -> Path:

    # Idempotent: creates data/notes/ if missing, does nothing if it already exists.
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    base_name = _sanitize_base_name(file_name)
    note_path = NOTES_DIR / f"{base_name}_notes.txt"

    # Every note is meant to be its own new file, so an existing file with the
    # same name is never overwritten -- disambiguate with a numeric suffix instead.
    suffix = 2
    while note_path.exists():
        note_path = NOTES_DIR / f"{base_name}_notes_{suffix}.txt"
        suffix += 1

    note_path.write_text(content, encoding="utf-8")

    return note_path
