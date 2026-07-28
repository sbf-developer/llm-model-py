# Append custom text or dialogue to data/data.txt

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_util import append_dialogue, append_text, data_stats
from scripts.build_training_data import main as rebuild_main


def main():
    parser = argparse.ArgumentParser(description="Add training data to data/data.txt")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="show current data file size")

    p_text = sub.add_parser("text", help="append a block of plain text")
    p_text.add_argument("content", nargs="?", help="text to append (or use --file)")
    p_text.add_argument("--file", "-f", help="read text from a file")
    p_text.add_argument("--label", help="section header in data.txt")

    p_dlg = sub.add_parser("dialogue", help="append one User/Assistant pair")
    p_dlg.add_argument("user", help="user message")
    p_dlg.add_argument("assistant", help="assistant reply")

    sub.add_parser("batch", help="append generated dialogue batch (--append)")
    sub.add_parser("rebuild", help="overwrite data.txt with full generated corpus")

    args = parser.parse_args()

    if args.cmd == "stats":
        s = data_stats()
        print(f"path: {s['path']}")
        print(f"chars: {s['chars']:,} | lines: {s['lines']:,}")
        return

    if args.cmd == "text":
        if args.file:
            content = Path(args.file).read_text(encoding="utf-8")
        elif args.content:
            content = args.content
        else:
            print("Provide text or --file", file=sys.stderr)
            sys.exit(1)
        result = append_text(content, label=args.label)
        print(f"Appended {result['appended_chars']:,} chars -> total {result['chars']:,}")

    elif args.cmd == "dialogue":
        result = append_dialogue(args.user, args.assistant)
        print(f"Appended dialogue -> total {result['chars']:,} chars")

    elif args.cmd == "batch":
        import sys as _sys
        _sys.argv = ["build_training_data.py", "--append"]
        rebuild_main()
        s = data_stats()
        print(f"Batch appended -> total {s['chars']:,} chars")

    elif args.cmd == "rebuild":
        rebuild_main()


if __name__ == "__main__":
    main()
