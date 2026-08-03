# Owns all file I/O and path constants for the orchestrator's per-session chat log.
# This is the durable, LLM-summarized record of the session (one entry per turn) --
# distinct from the raw, in-RAM turn history LangGraph's MemorySaver checkpoints
# under the session's thread_id (see graph.py/state.py). This module never feeds
# the per-turn prompts; it only powers the end-of-session note (see llm.py's
# orchestrator_mem_update) and the "bye" save-to-file flow.
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

ORCHESTRATOR_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = ORCHESTRATOR_DIR.parent.parent

CHAT_LOG_DIR = AGENTSYSTEM_ROOT / "data" / "chat_log"
CHAT_LOG_PATH = CHAT_LOG_DIR / "chat_log.md"

EMPTY_CHAT_LOG_PLACEHOLDER = "(no chat log yet)"

# Anchors on the exact header append_entry() writes: "Summary to question N: \n"
_ENTRY_HEADER_RE = re.compile(r"^Summary to question (\d+): *\n", re.MULTILINE)

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_SLUG_LENGTH = 50


# Splits file content into (question_number, raw_entry_text) tuples, in file order.
def _parse_entries(content: str) -> list[tuple[int, str]]:

    matches = list(_ENTRY_HEADER_RE.finditer(content))

    entries = []
    for i, m in enumerate(matches):
        # Each entry spans from this header's start up to the next header's
        # start (or EOF for the last one) -- tolerant of whatever the note body
        # itself contains (multi-line text, blank lines, etc.).
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        entries.append((int(m.group(1)), content[start:end]))

    return entries


# Turns arbitrary LLM-authored text into a safe, non-empty filename slug.
def _sanitize_slug(raw: str) -> str:

    # Only safe filesystem chars survive; everything else collapses to "_".
    slug = _SAFE_CHARS_RE.sub("_", raw.strip()).strip("_").lower()
    slug = slug[:_MAX_SLUG_LENGTH].strip("_")

    if not slug:
        # No usable slug was provided -- fall back to a timestamp so the save
        # still succeeds under a unique name.
        slug = datetime.now().strftime("chat_%Y%m%d_%H%M%S")

    return slug


# Creates data/chat_log/ and resets chat_log.md to empty; called once at boot.
def init_chat_log() -> None:

    # Reset on every boot (not just idempotent creation), so a crashed/un-"bye"'d
    # previous session can never silently bleed its leftover chat log into a new one.
    had_leftover = CHAT_LOG_PATH.exists() and CHAT_LOG_PATH.stat().st_size > 0

    CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_LOG_PATH.write_text("", encoding="utf-8")

    if had_leftover:
        print("Note: discarded unsaved chat log left over from a previous session.")


# Returns every chat log entry written so far, as one string.
def read_chat_log() -> str:

    if not CHAT_LOG_PATH.exists():
        return EMPTY_CHAT_LOG_PLACEHOLDER

    content = CHAT_LOG_PATH.read_text(encoding="utf-8")
    entries = _parse_entries(content)

    if not entries:
        return EMPTY_CHAT_LOG_PLACEHOLDER

    return "".join(text for _, text in entries).strip()


# Returns the highest question number already on disk (0 if none), for the next entry.
def count_existing_entries() -> int:

    if not CHAT_LOG_PATH.exists():
        return 0

    entries = _parse_entries(CHAT_LOG_PATH.read_text(encoding="utf-8"))
    return max((n for n, _ in entries), default=0)


# Appends one chat log entry in the exact "Summary to question N: \n<note>\n\n" format.
def append_entry(question_number: int, note_text: str) -> None:

    CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry = f"Summary to question {question_number}: \n{note_text}\n\n"
    with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)


# Ends the session: either discards the chat log file, or renames/moves it into data/.
# depending on the user's input
def finalize_and_save(keep: bool, new_name: Optional[str] = None) -> Optional[Path]:

    if not keep:
        # Discard: remove both the file and the now-unneeded working folder.
        if CHAT_LOG_PATH.exists():
            CHAT_LOG_PATH.unlink()
        if CHAT_LOG_DIR.exists():
            shutil.rmtree(CHAT_LOG_DIR)
        return None

    slug = _sanitize_slug(new_name or "")
    dest_dir = AGENTSYSTEM_ROOT / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / f"{slug}_chat_points.md"

    # Never overwrite a previous save with the same slug -- disambiguate with a
    # numeric suffix instead (same pattern as notesTool/noteWriter.py).
    suffix = 2
    while dest_path.exists():
        dest_path = dest_dir / f"{slug}_chat_points_{suffix}.md"
        suffix += 1

    if CHAT_LOG_PATH.exists():
        shutil.move(str(CHAT_LOG_PATH), str(dest_path))
    else:
        # Nothing was ever written this session (e.g. "bye" on the very first
        # turn) -- still produce a file rather than silently doing nothing.
        dest_path.write_text("(no chat log recorded this session)\n", encoding="utf-8")

    if CHAT_LOG_DIR.exists():
        shutil.rmtree(CHAT_LOG_DIR)

    return dest_path
