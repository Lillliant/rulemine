"""Unit tests for git evidence linking."""

from pathlib import Path
from rulemine.gitlink import find_nearby_commit


def test_find_nearby_commit_in_workspace():
    # Current repo is a git workspace
    commit = find_nearby_commit(Path("."))
    assert commit is not None
    assert len(commit) >= 7


def test_find_nearby_commit_non_git(tmp_path: Path):
    non_git = tmp_path / "empty_dir"
    non_git.mkdir()
    commit = find_nearby_commit(non_git)
    assert commit is None


def test_find_nearby_commit_invalid_inputs():
    assert find_nearby_commit("") is None
    assert find_nearby_commit("/nonexistent/path/here") is None
