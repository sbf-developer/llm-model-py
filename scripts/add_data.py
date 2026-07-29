# Append custom text or dialogue to data/data.txt

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_util import append_dialogue, append_text, data_stats
from scripts.build_training_data import build_wiki_corpus, main as rebuild_main
from scripts.stem_corpus import build_stem_corpus
from scripts.knowledge_corpus import build_knowledge_corpus
from scripts.natural_corpus import build_natural_corpus
from scripts.fetch_wikipedia import build_wikipedia_corpus
from scripts.fetch_web_corpus import build_web_corpus
from scripts.fetch_prose_corpus import build_prose_corpus
from scripts.fetch_short_corpus import build_short_corpus
from data_util import append_text as _append_text


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
    sub.add_parser("wiki", help="append encyclopedia articles and wiki-style Q&A")
    sub.add_parser("stem", help="append science, math, physics, and multi-turn chat data")
    sub.add_parser("knowledge", help="append wiki, psychology, stories, jokes, and life data")
    sub.add_parser("natural", help="append casual chat, story exchanges, and natural dialogue")
    sub.add_parser("wikipedia", help="fetch modern English summaries from Wikipedia API")
    sub.add_parser("web", help="fetch trivia, dictionary, poetry, wiki random, NASA, number facts")
    sub.add_parser("prose", help="fetch plain English paragraphs (wiki, science, poetry — no Q&A format)")
    sub.add_parser("chunks", help="fetch short trivia, dictionary, poetry, NASA mix (plain + chat)")
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

    elif args.cmd == "wiki":
        result = _append_text(build_wiki_corpus(), label="Wiki corpus")
        print(f"Wiki appended {result['appended_chars']:,} chars -> total {result['chars']:,}")

    elif args.cmd == "stem":
        result = _append_text(build_stem_corpus(), label="STEM corpus")
        print(f"STEM appended {result['appended_chars']:,} chars -> total {result['chars']:,}")

    elif args.cmd == "knowledge":
        result = _append_text(build_knowledge_corpus(), label="Knowledge corpus")
        print(f"Knowledge appended {result['appended_chars']:,} chars -> total {result['chars']:,}")

    elif args.cmd == "natural":
        result = _append_text(build_natural_corpus(), label="Natural conversation")
        print(f"Natural appended {result['appended_chars']:,} chars -> total {result['chars']:,}")

    elif args.cmd == "wikipedia":
        result = _append_text(build_wikipedia_corpus(), label="Wikipedia fetch")
        print(f"Wikipedia appended {result['appended_chars']:,} chars -> total {result['chars']:,}")

    elif args.cmd == "web":
        print("Fetching from free web APIs (may take 2-4 minutes)...")
        result = _append_text(build_web_corpus(), label="Web API fetch")
        print(f"Web corpus appended {result['appended_chars']:,} chars -> total {result['chars']:,}")

    elif args.cmd == "prose":
        print("Fetching plain prose (may take 3-5 minutes)...")
        result = _append_text(build_prose_corpus(), label="Plain prose")
        print(f"Prose appended {result['appended_chars']:,} chars -> total {result['chars']:,}")

    elif args.cmd == "chunks":
        print("Fetching short English chunks (trivia, dictionary, poetry, NASA)...")
        result = _append_text(build_short_corpus(), label="Short English chunks")
        print(f"Chunks appended {result['appended_chars']:,} chars -> total {result['chars']:,}")

    elif args.cmd == "rebuild":
        rebuild_main()


if __name__ == "__main__":
    main()
