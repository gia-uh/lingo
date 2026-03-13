#!/usr/bin/env python3
"""
Hook to enforce project journaling conditionally for all significant changes.

This hook is triggered 'AfterAgent' and checks if the daily journal entry has been updated
whenever there are NEW uncommitted changes since the last recorded journal update.
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# Add the hooks directory to path for importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

# Path to the state file relative to this script
STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "last_journal_update"
)


def get_last_update_timestamp():
    """Reads the last recorded journal update timestamp from the state file."""
    if not os.path.exists(STATE_FILE):
        return 0
    try:
        with open(STATE_FILE, "r") as f:
            return float(f.read().strip())
    except (ValueError, OSError):
        return 0


def update_last_update_timestamp():
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
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-uall"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.splitlines()
        return [line[3:].strip() for line in lines if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def main():
    """
    Main entry point for the journal enforcement hook.

    Identifies new uncommitted changes and identifies if the daily journal
    needs to be updated. Returns 'deny' if it's missing for new work.
    """
    try:
        changed_files = get_changed_files()

        today = datetime.now().strftime("%Y-%m-%d")
        journal_file = f"journal/{today}.md"

        last_update = get_last_update_timestamp()

        # 1. Identify files modified after last_update (excluding the journal)
        new_significant_changes = []
        latest_change_time = 0

        for f in changed_files:
            if f == journal_file:
                continue

            f_mtime = 0
            if os.path.exists(f):
                f_mtime = os.path.getmtime(f)
            else:
                # File deleted. We treat deletions as new work "now".
                f_mtime = time.time()

            if f_mtime > last_update:
                new_significant_changes.append(f)
                latest_change_time = max(latest_change_time, f_mtime)

        if not new_significant_changes:
            # No new work since last journal update.
            # However, if the journal itself was updated in the turn,
            # we should update the timestamp so we don't re-process old changes.
            if journal_file in changed_files:
                update_last_update_timestamp()
            utils.send_hook_decision("allow")
            return

        # 2. We have new significant changes. Check if the journal was updated AFTER them.
        if journal_file in changed_files:
            journal_mtime = os.path.getmtime(journal_file)
            if journal_mtime > latest_change_time:
                # Journal was updated after the latest significant change.
                update_last_update_timestamp()
                utils.send_hook_decision("allow")
                return

        # 3. Deny because new work exists but journal hasn't caught up.
        utils.send_hook_decision(
            "deny",
            reason=(
                f"New changes detected: {', '.join(new_significant_changes[:3])}{'...' if len(new_significant_changes) > 3 else ''}.\n"
                f"Please add or update a one-line entry to {journal_file} "
                "describing the work you just did. Do not stop until this file is updated."
            ),
        )

    except Exception as e:
        # Failsafe: always allow if something goes wrong
        utils.send_hook_decision("allow", reason=f"Hook error: {str(e)}")


if __name__ == "__main__":
    main()
