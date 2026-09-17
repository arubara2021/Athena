from pathlib import Path

# ==========================
# CONFIG
# ==========================

PROJECT_ROOT = Path(".").resolve()

OUTPUT_DIR = PROJECT_ROOT / "study_exports"
OUTPUT_DIR.mkdir(exist_ok=True)

NUMBER_OF_PARTS = 4

# File types to include
INCLUDE_EXTENSIONS = {
    ".py",
    
    ".yaml",
    ".yml",
    
}

# Folders to ignore
EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "logs",
    "output",
    "data",
    "study_exports",
    ".json",
    ".txt",
}

# Files to ignore
EXCLUDE_FILES = {
    "research_agent.db",
    "vector_store.json",
}


# ==========================
# HELPERS
# ==========================

def should_include_file(path: Path) -> bool:
    """
    Decide whether a file should be included in the exported study files.
    """

    # Ignore excluded folders
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False

    # Ignore excluded files
    if path.name in EXCLUDE_FILES:
        return False

    # Ignore generated export files
    if path.name.startswith("project_part_") and path.suffix == ".txt":
        return False

    # Include only selected extensions
    if path.suffix.lower() not in INCLUDE_EXTENSIONS:
        return False

    return True


def read_file_safely(path: Path) -> str:
    """
    Read text file safely.
    If encoding is strange, replace unreadable characters.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR READING FILE: {e}]"


def make_file_block(path: Path) -> str:
    """
    Create a clean text block for one file.
    """
    relative_path = path.relative_to(PROJECT_ROOT)
    content = read_file_safely(path)

    return f"""
{'=' * 100}
FILE: {relative_path}
{'=' * 100}

{content}

"""


# ==========================
# MAIN EXPORT LOGIC
# ==========================

def main():
    files = []

    for path in PROJECT_ROOT.rglob("*"):
        if path.is_file() and should_include_file(path):
            files.append(path)

    # Sort files so the study order is stable and readable
    files.sort(key=lambda p: str(p.relative_to(PROJECT_ROOT)).lower())

    if not files:
        print("No files found to export.")
        return

    # Build blocks first
    blocks = []
    total_chars = 0

    for file_path in files:
        block = make_file_block(file_path)
        blocks.append((file_path, block))
        total_chars += len(block)

    target_size = total_chars / NUMBER_OF_PARTS

    parts = [[] for _ in range(NUMBER_OF_PARTS)]
    part_sizes = [0 for _ in range(NUMBER_OF_PARTS)]

    current_part = 0

    for file_path, block in blocks:
        # Move to next part if current part is already near target
        # Keep whole files together, do not split a file in the middle.
        if (
            current_part < NUMBER_OF_PARTS - 1
            and part_sizes[current_part] + len(block) > target_size
            and part_sizes[current_part] > 0
        ):
            current_part += 1

        parts[current_part].append(block)
        part_sizes[current_part] += len(block)

    # Write output files
    for i, part_blocks in enumerate(parts, start=1):
        output_file = OUTPUT_DIR / f"project_part_{i}.txt"

        header = f"""
PROJECT STUDY EXPORT
PART {i} OF {NUMBER_OF_PARTS}

Root folder:
{PROJECT_ROOT}

This file contains exported project files for reading/studying.

"""

        output_file.write_text(
            header + "\n".join(part_blocks),
            encoding="utf-8"
        )

    # Print summary
    print("Export complete.")
    print(f"Output folder: {OUTPUT_DIR}")
    print()

    for i, size in enumerate(part_sizes, start=1):
        print(f"project_part_{i}.txt -> {size:,} characters")

    print()
    print("Files included:")
    for file_path in files:
        print("-", file_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()