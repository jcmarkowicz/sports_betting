
import subprocess

import sys
import os 


# Function to commit if changed
def commit_if_changed(file_path, msg, branch="main"):

    repo = os.environ["GITHUB_REPOSITORY"]  # e.g., 'username/repo'
    token = os.environ["GITHUB_TOKEN"]      # Provided automatically in Actions

    # Stage file
    subprocess.run(["git", "add", str(file_path)], check=True)

    # Check if staged changes exist
    diff_check = subprocess.run(
        ["git", "diff", "--cached", "--quiet", str(file_path)]
    )

    if diff_check.returncode != 0:
        # Configure git
        subprocess.run(["git", "config", "--global", "user.name", "github-actions[bot]"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)

        # Commit changes
        subprocess.run(["git", "commit", "-m", msg], check=True)

        # Push using token authentication
        push_url = f"https://x-access-token:{token}@github.com/{repo}.git"
        subprocess.run(["git", "push", push_url, f"HEAD:{branch}"], check=True)
