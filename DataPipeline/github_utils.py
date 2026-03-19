
import subprocess
from io import StringIO

import sys
import os 

import numpy as np 
import pandas as pd


# Function to commit if changed
def commit_if_changed(file_path, msg, branch="main"):

    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]

    if not floats_changed(file_path):
        return

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


def floats_changed(file_path, tol=2e-10):
    # current file
    df_new = pd.read_csv(file_path)

    # previous committed version
    try:
        old_bytes = subprocess.check_output(["git", "show", f"HEAD:{file_path}"])
    except subprocess.CalledProcessError:
        return True  # file not previously tracked

    df_old = pd.read_csv(StringIO(old_bytes.decode()))

    if df_new.shape != df_old.shape:
        return True

    for col in df_new.columns:
        a = df_new[col]
        b = df_old[col]

        if np.issubdtype(a.dtype, np.number):
            if not np.allclose(a.values, b.values, atol=tol, rtol=0):
                return True
        else:
            if not a.equals(b):
                return True

    return False