import subprocess
import sys


def main() -> int:
    commands = [
        ["uv", "run", "pytest"],
        ["uv", "run", "python", "-m", "compileall", "src", "tests"],
        ["uv", "build"],
    ]
    for command in commands:
        print(f"$ {' '.join(command)}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
