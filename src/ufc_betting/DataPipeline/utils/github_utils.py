
import os 
import subprocess
from pathlib import Path

import numpy as np 
import pandas as pd

from pandas.api.types import is_float_dtype, is_integer_dtype

"""
Subprocess library to run shell commands for git operations. 
"""

def push_with_sync(branch="main", attempts=3):
    """Preserve remote commits before publishing generated results."""
    # actions/checkout supplies origin authentication; keep tokens out of commands.
    for attempt in range(attempts):
        subprocess.run(["git", "fetch", "origin", branch], check=True)
        rebase = subprocess.run(["git", "rebase", "FETCH_HEAD"], check=False)
        if rebase.returncode:
            subprocess.run(["git", "rebase", "--abort"], check=False)
            raise RuntimeError(
                "Cannot publish generated results: rebasing onto GitHub failed. "
                "Check the Git output above for conflicts or uncommitted changes; "
                "no force push was attempted."
            )
        push = subprocess.run(
            ["git", "push", "origin", f"HEAD:{branch}"],
            capture_output=True, text=True, check=False,
        )
        if push.returncode == 0:
            return
        detail = (push.stdout or "") + (push.stderr or "")
        remote_advanced = "[rejected]" in detail and any(
            reason in detail for reason in ("fetch first", "non-fast-forward")
        )
        if not remote_advanced or attempt == attempts - 1:
            raise RuntimeError(f"Git push failed: {detail.strip()}")
        print("GitHub advanced during publishing; fetching and rebasing before retry.")


# Function to commit if changed
def commit_if_changed(df, file_path, msg, branch="main"):


    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not numeric_changed(df, file_path):
        return

    df = df.reset_index(drop=True)
    df.index = range(len(df))
    df.to_csv(file_path, index=False)

    subprocess.run(["git", "add", str(file_path)], check=True)

    result = subprocess.run(
    ["git", "diff", "--cached", "--quiet", str(file_path)]
)
    if result.returncode == 0:
        # No changes staged → skip commit
        return
    
    subprocess.run(
        ["git", "config", "--global", "user.name", "github-actions[bot]"],
        check=True,
    )
    subprocess.run(
        ["git", "config", "--global", "user.email", "github-actions[bot]@users.noreply.github.com"],
        check=True,
    )

    subprocess.run(["git", "commit", "-m", msg], check=True)

    push_with_sync(branch)


def numeric_changed(df_new, file_path, tol=2e-3):

    if not os.path.exists(file_path):
        return True

    df_old = pd.read_csv(file_path)

    if df_new.shape != df_old.shape:
        return True

    for col in df_new.columns:
        a = df_new[col]
        b = df_old[col]

        if is_float_dtype(a):
            if not np.allclose(a.to_numpy(), b.to_numpy(), atol=tol, rtol=0, equal_nan=True):
                return True

        elif is_integer_dtype(a):
            if not np.array_equal(a.to_numpy(), b.to_numpy()):
                return True

        else:
            if not a.equals(b):
                return True

    return False

def delete_and_commit(path, message):

    file_path = Path(path)

    if not file_path.exists():
        return

    # 1. Delete the file
    file_path.unlink(missing_ok=True)

    # 2. Stage the deletion (IMPORTANT)
    subprocess.run(["git", "rm", str(file_path)], check=False)

    # 3. Commit the deletion
    subprocess.run(["git", "commit", "-m", message], check=False)

    # 4. Push
    push_with_sync()


def commit_figure_if_changed(
    file_path: str | Path,
    message: str | None = None,
    branch: str = "main",
) -> bool:
    """Commit and push a saved figure when its file contents changed."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Figure does not exist: {file_path}")

    subprocess.run(["git", "add", str(file_path)], check=True)

    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", str(file_path)],
        check=False,
    )
    if diff.returncode == 0:
        return False
    if diff.returncode != 1:
        raise subprocess.CalledProcessError(diff.returncode, diff.args)

    subprocess.run(
        ["git", "config", "--global", "user.name", "github-actions[bot]"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "--global",
            "user.email",
            "github-actions[bot]@users.noreply.github.com",
        ],
        check=True,
    )

    commit_message = message or f"Updating {file_path.name}"
    subprocess.run(
        ["git", "commit", "-m", commit_message, "--", str(file_path)],
        check=True,
    )

    push_with_sync(branch)
    return True
