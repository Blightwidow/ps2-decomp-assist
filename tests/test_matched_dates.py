import subprocess
from pathlib import Path

from pipeline.matched_dates import earliest_date, first_seen_dates


def _git(repository: Path, *arguments: str, date: str = "2026-01-01T00:00:00") -> None:
    environment = {
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
        "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
        "HOME": str(repository),
    }
    subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True, env=environment)


def test_first_seen_date_is_earliest_commit_adding_the_name(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.cpp").write_text("int first(int a) { return a; }\n")
    _git(tmp_path, "add", "src/a.cpp")
    _git(tmp_path, "commit", "-q", "-m", "one", date="2026-02-03T12:00:00")
    (tmp_path / "src" / "a.cpp").write_text("int first(int a) { return a; }\nint second(int a) { return a; }\n")
    _git(tmp_path, "add", "src/a.cpp")
    _git(tmp_path, "commit", "-q", "-m", "two", date="2026-03-04T12:00:00")
    dates = first_seen_dates(tmp_path, "src")
    assert (dates["first"], dates["second"]) == ("2026-02-03", "2026-03-04")


def test_earliest_date_takes_minimum_across_candidates() -> None:
    assert earliest_date({"readable", "func_1"}, {"readable": "2026-03-01", "func_1": "2026-02-01"}) == "2026-02-01"
    assert earliest_date({"missing"}, {}) is None
