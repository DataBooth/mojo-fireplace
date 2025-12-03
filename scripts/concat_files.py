#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python concat.py <directory>", file=sys.stderr)
        sys.exit(1)

    dir_path = Path(sys.argv[1]).resolve()
    if not dir_path.is_dir():
        print(f"Error: '{dir_path}' is not a directory", file=sys.stderr)
        sys.exit(1)

    output_file = dir_path / "all.txt"

    with open(output_file, "w") as out:  # 'w' mode overwrites existing file
        for path in sorted(dir_path.iterdir()):
            if (
                path.is_file()
                and not path.is_symlink()
                and not path.suffix.lower() == ".off"
            ):
                print(f"===== {path.name} =====", file=out)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        out.write(f.read())
                except Exception:
                    pass  # Skip unreadable files silently
                out.write("\n\n")

    print(f"Concatenated to {output_file}")


if __name__ == "__main__":
    main()
