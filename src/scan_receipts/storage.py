from __future__ import annotations

import shutil
from pathlib import Path

from send2trash import send2trash


def delete_session_video(
    video_path: str | Path, video_root: str | Path, to_trash: bool = False
) -> Path:
    """Delete one session video without permitting arbitrary directory deletion."""
    path = Path(video_path).resolve()
    root = Path(video_root).resolve()

    if path == root:
        raise RuntimeError("Refusing to delete the configured video root")

    if path.is_dir():
        if root not in path.parents:
            raise RuntimeError(
                "Refusing to recursively delete an external directory; "
                "the replay source must be a video file"
            )
        if to_trash:
            send2trash(str(path))
        else:
            shutil.rmtree(path)
    elif path.is_file():
        if to_trash:
            send2trash(str(path))
        else:
            path.unlink()

    return path
