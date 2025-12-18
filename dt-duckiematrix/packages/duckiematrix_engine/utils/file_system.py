"""File system."""

import shutil
from pathlib import Path


def get_owner_gid(fpath: str) -> int:
    """Return owner gid."""
    return Path(fpath).stat().st_gid.real


def get_owner_uid(fpath: str) -> int:
    """Return owner uid."""
    return Path(fpath).stat().st_uid.real


def get_ownership(fpath: str) -> tuple[int, int]:
    """Return owner uid and gid."""
    return get_owner_uid(fpath), get_owner_gid(fpath)


def set_owner(fpath: str, uid: int, gid: int) -> None:
    """Set owner."""
    shutil.chown(fpath, uid, gid)


def set_ownership(
    fpath: str,
    uid: int,
    gid: int,
    *,
    recursive: bool = False,
) -> None:
    """Set ownership."""
    set_owner(fpath, uid, gid)
    if recursive:
        files = Path(fpath).rglob("*")
        for file in files:
            set_owner(str(file), uid, gid)
