"""Short modern English chunks: trivia, dictionary, poetry, NASA.

Best mix for a tiny char-level model — varied prose in bite-sized pieces,
plus some User/Assistant pairs so chat still works.
"""

from __future__ import annotations

from scripts.build_training_data import section
from scripts.fetch_prose_corpus import (
    fetch_dictionary_prose,
    fetch_nasa_prose,
    fetch_poetry_prose,
    fetch_trivia_prose,
)
from scripts.fetch_web_corpus import (
    COMMON_WORDS,
    fetch_dictionary,
    fetch_nasa_apod,
    fetch_poetry,
    fetch_trivia,
)

# words not covered by the default dictionary fetch lists
EXTRA_WORDS = [
    "wonder", "believe", "create", "explore", "gentle", "clever", "silent",
    "rapid", "ancient", "modern", "strange", "familiar", "delicate", "powerful",
    "modest", "brilliant", "anxious", "calm", "eager", "grateful", "honest",
    "loyal", "mystery", "journey", "moment", "silence", "rhythm", "balance",
    "harmony", "texture", "pattern", "structure", "element", "reaction",
    "gravity", "orbit", "spectrum", "microbe", "organism", "habitat", "climate",
    "volcano", "glacier", "meadow", "harbor", "frontier", "village", "library",
    "theater", "sculpture", "melody", "chapter", "sentence", "paragraph",
]


def build_short_corpus(
    *,
    trivia_count: int = 100,
    dictionary_words: list[str] | None = None,
    poetry_count: int = 35,
    nasa_count: int = 25,
    include_qa: bool = True,
) -> str:
    """Fetch trivia + dictionary + poetry + NASA as short plain chunks (+ optional Q&A)."""
    dictionary_words = dictionary_words or list(dict.fromkeys(COMMON_WORDS + EXTRA_WORDS))
    parts: list[str] = []

    plain_fetchers = [
        ("trivia (plain)", lambda: fetch_trivia_prose(trivia_count)),
        ("dictionary (plain)", lambda: fetch_dictionary_prose(dictionary_words)),
        ("poetry (plain)", lambda: fetch_poetry_prose(poetry_count, max_lines=6)),
        ("nasa (plain)", lambda: fetch_nasa_prose(nasa_count)),
    ]
    for name, fn in plain_fetchers:
        try:
            block = fn()
            if block:
                parts.append(block)
                print(f"  + {name}: {len(block):,} chars")
        except Exception as exc:
            print(f"  ! {name} skipped: {exc}")

    if include_qa:
        qa_fetchers = [
            ("trivia (chat)", lambda: fetch_trivia(min(60, trivia_count))),
            ("dictionary (chat)", lambda: fetch_dictionary(dictionary_words)),
            ("poetry (chat)", lambda: fetch_poetry(min(20, poetry_count), max_lines=6)),
            ("nasa (chat)", lambda: fetch_nasa_apod(min(15, nasa_count))),
        ]
        for name, fn in qa_fetchers:
            try:
                block = fn()
                if block:
                    parts.append(block)
                    print(f"  + {name}: {len(block):,} chars")
            except Exception as exc:
                print(f"  ! {name} skipped: {exc}")

    if not parts:
        raise RuntimeError("No short-chunk content fetched. Check your internet connection.")

    body = "\n".join(parts)
    return section("Short English chunks", body)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch short trivia/dictionary/poetry/NASA chunks for training"
    )
    parser.add_argument("--print", action="store_true", help="print corpus to stdout")
    parser.add_argument("--plain-only", action="store_true", help="skip User/Assistant pairs")
    args = parser.parse_args()

    print("Fetching short English chunks...")
    corpus = build_short_corpus(include_qa=not args.plain_only)
    print(f"Total: {len(corpus):,} chars")
    if args.print:
        print(corpus)


if __name__ == "__main__":
    main()
