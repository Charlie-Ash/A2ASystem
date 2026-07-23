# Shared constants for the notes tool.
from pathlib import Path

# Resolved relative to this file rather than the process's CWD (same pattern as ragTool/config.py).
NOTESTOOL_DIR = Path(__file__).resolve().parent
AGENTSYSTEM_ROOT = NOTESTOOL_DIR.parent.parent.parent

# Notes are written into the project's shared top-level data/ folder, in their own subfolder.
NOTES_DIR = AGENTSYSTEM_ROOT / "data" / "notes"
