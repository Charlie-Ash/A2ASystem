# Owns all file I/O and path constants for the orchestrator's per-session chat log.
# This is the durable, LLM-summarized record of the session (one entry per turn) --
# distinct from the raw, in-RAM turn history LangGraph's MemorySaver checkpoints
# under the session's thread_id (see graph.py/state.py). This module never feeds
# the per-turn prompts; it only powers the end-of-session note (see llm.py's
# orchestrator_mem_update) and the "bye" save-to-file flow.
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

ORCHESTRATOR_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = ORCHESTRATOR_DIR.parent.parent

CHAT_LOG_DIR = AGENTSYSTEM_ROOT / "data" / "chat_log"
CHAT_LOG_PATH = CHAT_LOG_DIR / "chat_log.md"

EMPTY_CHAT_LOG_PLACEHOLDER = "(no chat log yet)"

# Where the current session's thread_id is persisted, so a restarted
# orchestrator process can resume the same LangGraph checkpointed state
# (see orchestrator.py's AsyncSqliteSaver) instead of opening a fresh, empty
# thread. Same data/-directory convention as CHAT_LOG_DIR above, computed
# independently rather than importing orchestrator/config.py's
# CHECKPOINT_DB_PATH -- if that env var is ever overridden to a different
# directory, this file won't follow it; accepted as a minor drift risk for a
# solo, short-timeline project rather than adding a cross-module import.
CHECKPOINT_DIR = AGENTSYSTEM_ROOT / "data" / "checkpoints"
THREAD_ID_PATH = CHECKPOINT_DIR / "thread_id.txt"

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


# Reads back a thread_id persisted by a previous, unclean (crashed/killed)
# session, or creates and persists a fresh one -- called once at boot,
# mirroring init_chat_log()'s reset-on-boot role for the markdown log.
# Unlike that log, a leftover thread_id is *resumed*, not discarded: it's
# what lets a restarted orchestrator process find its own prior checkpointed
# conversation state again in the persistent AsyncSqliteSaver (see
# orchestrator.py) instead of starting a new, empty thread.
def load_or_create_thread_id() -> str:

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if THREAD_ID_PATH.exists():
        existing = THREAD_ID_PATH.read_text(encoding="utf-8").strip()
        if existing:
            print(f"Note: resuming previous session's thread ({existing}) after an unclean exit.")
            return existing

    thread_id = str(uuid.uuid4())
    THREAD_ID_PATH.write_text(thread_id, encoding="utf-8")
    return thread_id


# Deletes the persisted thread_id -- called on a clean "bye" exit (see
# Orchestrator.end_session) so the *next* process boot starts a genuinely
# new session/thread, rather than resuming a conversation the user already
# ended on purpose.
def clear_thread_id() -> None:

    if THREAD_ID_PATH.exists():
        THREAD_ID_PATH.unlink()


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
    # numeric suffix instead (same pattern as actionsTool/actions/noteWriter.py).
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
