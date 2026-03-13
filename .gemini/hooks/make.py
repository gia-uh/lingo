#!/usr/bin/env python3
"""
Hook to run 'make' validation conditionally before critical agent actions.

This hook is triggered 'AfterAgent' and ensures that the codebase passes
all linting and testing checks defined in the makefile. It skips execution
if no changes have been detected since the last successful run.
"""

import sys
import os
import subprocess
import time

# Add the hooks directory to path for importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

# Path to the state file relative to this script
# Stored in .gemini/last_make_run (one level up from .gemini/hooks/)
STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "last_make_run"
)


def get_last_run_timestamp():
    """Reads the last successful run timestamp from the state file."""
    if not os.path.exists(STATE_FILE):
        return 0
    try:
        with open(STATE_FILE, "r") as f:
            return float(f.read().strip())
    except (ValueError, OSError):
        return 0


def update_last_run_timestamp():
    """Updates the state file with the current timestamp."""
    try:
        with open(STATE_FILE, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def get_changed_files():
    """
    Returns a list of files that are either modified or untracked in git.
    """
    try:
        # Get both modified and untracked files
        # --porcelain=v1 ensures a stable machine-readable output
        # -u (all) shows untracked files
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-uall"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.splitlines()
        # Each line is: "XY PATH" where XY are status codes
        # We want the path part (from index 3 onwards)
        return [line[3:].strip() for line in lines if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def should_run_make():
    """
    Determines if 'make' should be executed based on file modification times.
    """
    last_run = get_last_run_timestamp()
    if last_run == 0:
        return True, "Initial run or missing timestamp."

    changed_paths = get_changed_files()
    if not changed_paths:
        return False, "No uncommitted or untracked changes detected."

    for path in changed_paths:
        # Check if file exists (might have been deleted)
        if os.path.exists(path):
            if os.path.getmtime(path) > last_run:
                return True, f"File modified: {path}"
        else:
            # File was deleted. We treat deletions as changes.
            # Since the file is gone, its parent directory's mtime should have updated.
            # Or we can just assume any deletion warrants a re-run.
            return True, f"File deleted: {path}"

    return False, "No files modified since last successful validation."


def main():
    """
    Main entry point for the make validation hook.

    Executes 'make' conditionally and returns a decision.
    """
    try:
        run_needed, reason = should_run_make()

        if not run_needed:
            utils.send_hook_decision("allow", reason=f"Skipping 'make' ({reason})")
            return

        # Run make command directly
        result = subprocess.run(["make"], capture_output=True, text=True, check=False)

        if result.returncode != 0:
            # make failed
            error_message = result.stdout + "\n" + result.stderr
            fail_reason = (
                f"Validation failed (make returned {result.returncode}).\n"
                "Please fix the broken tests or linting issues.\n"
                "Output of 'make':\n"
                "```\n" + error_message.strip() + "\n```\n"
                "Fix these issues and ensure 'make' passes before continuing."
            )
            utils.send_hook_decision("deny", reason=fail_reason)
        else:
            # make passed
            update_last_run_timestamp()
            utils.send_hook_decision("allow")

    except Exception as e:
        # Failsafe: always allow if the hook itself fails, but log the error in reason
        utils.send_hook_decision("allow", reason=f"Hook error: {str(e)}")


if __name__ == "__main__":
    main()
