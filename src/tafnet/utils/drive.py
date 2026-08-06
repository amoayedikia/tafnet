"""
Google Drive helpers for GCP VM workflows.

On a Compute Engine VM the recommended way to access Google Drive is via
`rclone mount`. This module provides a small sanity check that fails loudly
if the configured drive_root does not exist or is empty, plus a helper to
build paths under the mount.

See docs/gcp_setup.md for full setup instructions.
"""
from __future__ import annotations

import os
from pathlib import Path


def verify_drive_mount(drive_root: str | os.PathLike) -> Path:
    """
    Verify Google Drive is mounted and contains files.

    Raises FileNotFoundError if the mount point doesn't exist or is empty,
    which would otherwise produce confusing 'file not found' errors deep
    inside the data loader.
    """
    root = Path(drive_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Drive mount not found at {root}.\n"
            "Mount your Google Drive with rclone first:\n"
            "  rclone mount mydrive: /mnt/drive --daemon --vfs-cache-mode writes\n"
            "See docs/gcp_setup.md for full setup."
        )
    try:
        entries = list(root.iterdir())
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot list {root} — check rclone mount permissions: {exc}"
        ) from exc
    if not entries:
        raise FileNotFoundError(
            f"Drive mount at {root} is empty. The mount may not be active. "
            "Try: `fusermount -u /mnt/drive && rclone mount mydrive: /mnt/drive ...`"
        )
    return root


def require_path(path: str | os.PathLike, what: str = "path") -> Path:
    """Assert a file or directory exists, with an informative error."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p
