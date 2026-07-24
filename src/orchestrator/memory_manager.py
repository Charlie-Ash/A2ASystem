# Owns all file I/O and path constants for the orchestrator's per-session conversation memory.
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

ORCHESTRATOR_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = ORCHESTRATOR_DIR.parent.parent

SYSTEM_MEM_DIR = AGENTSYSTEM_ROOT / "data" / "system_mem"
SYSTEM_MEMORY_PATH = SYSTEM_MEM_DIR / "system_memory.md"

# Cap on how many past entries get read back into the phase-1/phase-2 prompts,
# so a long session's memory can't blow the orchestrator's 4096-token budget.
MAX_MEMORY_ENTRIES = 8

EMPTY_MEMORY_PLACEHOLDER = "(no memory yet)"

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


# Creates data/system_mem/ and resets system_memory.md to empty; called once at boot.
def init_system_memory() -> None:

    # Reset on every boot (not just idempotent creation), so a crashed/un-"bye"'d
    # previous session can never silently bleed its leftover memory into a new one.
    had_leftover = SYSTEM_MEMORY_PATH.exists() and SYSTEM_MEMORY_PATH.stat().st_size > 0

    SYSTEM_MEM_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEM_MEMORY_PATH.write_text("", encoding="utf-8")

    if had_leftover:
        print("Note: discarded unsaved memory left over from a previous session.")


# Returns the last `max_entries` memory entries (or all, if None) as one string.
def read_system_memory(max_entries: Optional[int] = MAX_MEMORY_ENTRIES) -> str:

    if not SYSTEM_MEMORY_PATH.exists():
        return EMPTY_MEMORY_PLACEHOLDER

    content = SYSTEM_MEMORY_PATH.read_text(encoding="utf-8")
    entries = _parse_entries(content)

    if not entries:
        return EMPTY_MEMORY_PLACEHOLDER

    # Cap applies only to what gets shown to the LLM, never to what's on disk.
    if max_entries is not None:
        entries = entries[-max_entries:]

    return "".join(text for _, text in entries).strip()


# Returns the highest question number already on disk (0 if none), for the next entry.
def count_existing_entries() -> int:

    # Always parses the full, uncapped file -- this determines the next
    # question number, so it must never be affected by read_system_memory()'s cap.
    if not SYSTEM_MEMORY_PATH.exists():
        return 0

    entries = _parse_entries(SYSTEM_MEMORY_PATH.read_text(encoding="utf-8"))
    return max((n for n, _ in entries), default=0)


# Appends one memory entry in the exact "Summary to question N: \n<note>\n\n" format.
def append_entry(question_number: int, note_text: str) -> None:

    SYSTEM_MEM_DIR.mkdir(parents=True, exist_ok=True)

    entry = f"Summary to question {question_number}: \n{note_text}\n\n"
    with SYSTEM_MEMORY_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)


# Ends the session: either discards the memory file, or renames/moves it into data/.
# depending on the user's input
def finalize_and_save(keep: bool, new_name: Optional[str] = None) -> Optional[Path]:

    if not keep:
        # Discard: remove both the file and the now-unneeded working folder.
        if SYSTEM_MEMORY_PATH.exists():
            SYSTEM_MEMORY_PATH.unlink()
        if SYSTEM_MEM_DIR.exists():
            shutil.rmtree(SYSTEM_MEM_DIR)
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

    if SYSTEM_MEMORY_PATH.exists():
        shutil.move(str(SYSTEM_MEMORY_PATH), str(dest_path))
    else:
        # Nothing was ever written this session (e.g. "bye" on the very first
        # turn) -- still produce a file rather than silently doing nothing.
        dest_path.write_text("(no memory recorded this session)\n", encoding="utf-8")

    if SYSTEM_MEM_DIR.exists():
        shutil.rmtree(SYSTEM_MEM_DIR)

    return dest_path
