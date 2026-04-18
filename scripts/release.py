import subprocess
import sys


def run(command: list[str]) -> str:
    print(f"$ {' '.join(command)}")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.stdout.strip()


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"major", "minor", "patch"}:
        print("Usage: uv run python scripts/release.py <major|minor|patch>", file=sys.stderr)
        return 2

    bump = argv[1]
    branch = run(["git", "branch", "--show-current"])
    if branch != "main":
        print("Error: release must be run from the main branch.", file=sys.stderr)
        return 2

    status = run(["git", "status", "--short"])
    if status:
        print("Error: working tree must be clean before release.", file=sys.stderr)
        return 2

    run(["git", "pull", "--ff-only", "origin", "main"])
    version_output = run(["uv", "version", "--bump", bump])
    new_version = version_output.split()[-1]

    for command in (
        ["uv", "run", "pytest"],
        ["uv", "run", "python", "-m", "compileall", "src", "tests"],
        ["uv", "build"],
    ):
        run(command)

    run(["git", "add", "pyproject.toml", "uv.lock"])
    run(["git", "commit", "-m", f"chore(release): bump version to v{new_version}"])
    run(["git", "tag", "-a", f"v{new_version}", "-m", f"v{new_version}"])
    run(["git", "push", "origin", "main"])
    run(["git", "push", "origin", f"v{new_version}"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
