from __future__ import annotations
# Root conftest — makes the agentsnap pytest plugin auto-discoverable
# and establishes __agent_snapshots__/ at the project root.

# Prevent pytest from treating test_judge_connection in setup_wizard.py as a test.
collect_ignore = ["agentsnap/setup_wizard.py"]
