
import subprocess
from io import StringIO

import sys
import os 

import numpy as np 
import pandas as pd

from pathlib import Path

from pandas.api.types import is_numeric_dtype

# Function to commit if changed
def commit_if_changed(df, file_path, msg, branch="main"):

    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]

    if not floats_changed(df, file_path):
        return

    df = df.reset_index(drop=True)
    df.index = range(len(df))
    df.to_csv(file_path)

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

    push_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    subprocess.run(["git", "push", push_url, f"HEAD:{branch}"], check=True)


def floats_changed(df_new, file_path, tol=2e-3):
    
    # current file
    if not os.path.exists(file_path):
        return True

    df_old = pd.read_csv(file_path)
    if df_new.shape != df_old.shape:
        return True

    for col in df_new.columns:
        a = df_new[col]
        b = df_old[col]

    if is_numeric_dtype(a):
        if not np.allclose(a.to_numpy(), b.to_numpy(), atol=tol, rtol=0, equal_nan=True):
            return True
        
        else:
            if not a.equals(b):
                return True

    return False

def delete_and_commit(path, message):

    file_path = Path(path)

    # 1. Delete the file
    file_path.unlink(missing_ok=True)

    # 2. Stage the deletion (IMPORTANT)
    subprocess.run(["git", "rm", str(file_path)], check=False)

    # 3. Commit the deletion
    subprocess.run(["git", "commit", "-m", message], check=False)

    # 4. Push
    subprocess.run(["git", "push"], check=False)