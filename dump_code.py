import os
from datetime import datetime
FILES = [
    # Consensus ranker crash (_enum_value bug)
    "ranking/consensus_ranker.py",
    "ranking/llm_ranker.py",
    "core/models.py",              # Difficulty enum lives here
    "core/schemas.py",
    "llm/ensemble.py",
    "llm/aggregator.py",
    "llm/guardrails.py",
    "llm/parser.py",
    "llm/provider.py",

    # Dedup failure (same paper ranked #1, #2, #4)
    "search/dedupe.py",
    "search/normalizer.py",
    "utils/hashing.py",
]

OUTPUT_FILE = "code_dump.txt"
SEPARATOR = "=" * 100


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), None
    except FileNotFoundError:
        return None, "FILE NOT FOUND"
    except Exception as e:
        return None, f"ERROR: {e}"


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    found = []
    missing = []
    sections = []

    for rel_path in FILES:
        full_path = os.path.join(root, *rel_path.split("/"))
        content, error = read_file(full_path)
        if content is not None:
            found.append(rel_path)
            line_count = len(content.splitlines())
            header = (
                "\n\n" + SEPARATOR + "\n"
                + f"FILE: {rel_path}\n"
                + f"LINES: {line_count}\n"
                + f"CHARS: {len(content)}\n"
                + SEPARATOR + "\n\n"
            )
            sections.append(header + content.rstrip() + "\n")
        else:
            missing.append(f"{rel_path} -> {error}")

    summary_lines = [
        SEPARATOR,
        "CODE DUMP SUMMARY",
        f"GENERATED: {datetime.now().isoformat()}",
        f"TOTAL FILES REQUESTED: {len(FILES)}",
        f"FILES FOUND: {len(found)}",
        f"FILES MISSING: {len(missing)}",
        SEPARATOR,
        "",
        "FOUND FILES:",
    ]
    for f in found:
        summary_lines.append(f"  [OK] {f}")
    if missing:
        summary_lines.append("")
        summary_lines.append("MISSING / FAILED FILES:")
        for m in missing:
            summary_lines.append(f"  [XX] {m}")
    summary_lines.append(SEPARATOR)

    output_path = os.path.join(root, OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(summary_lines) + "\n")
        out.write("".join(sections))

    print(f"Done: {len(found)} files exported to {OUTPUT_FILE}")
    if missing:
        print(f"Warning: {len(missing)} files missing:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()