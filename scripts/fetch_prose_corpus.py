"""Fetch plain modern English prose (no User/Assistant formatting)."""

from __future__ import annotations

import html
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from scripts.build_training_data import section
from scripts.fetch_wikipedia import DEFAULT_TOPICS

USER_AGENT = "MiniLLM-TrainingBot/1.0 (local learning project)"

EXTRA_TOPICS = [
    "History", "Geography", "Literature", "Technology", "Medicine", "Agriculture",
    "Transport", "Communication", "Weather", "Volcano", "Dinosaur", "Galaxy",
    "Telescope", "Microscope", "Vaccine", "Democracy", "Constitution", "Painting",
    "Sculpture", "Theatre", "Basketball", "Tennis", "Cooking", "Bread", "Tea",
    "Chocolate", "Horse", "Whale", "Bee", "Tree", "Flower", "Desert", "Island",
    "London", "New York City", "Berlin", "Sydney", "Cairo", "Mexico City",
    "Albert Einstein", "Marie Curie", "Charles Darwin", "Leonardo da Vinci",
    "Industrial Revolution", "Renaissance", "Cold War", "Space exploration",
    "Renewable energy", "Recycling", "Biodiversity", "Ecosystem", "Meteorology",
]

COMMON_WORDS = [
    "beautiful", "important", "different", "possible", "interesting", "difficult",
    "remember", "understand", "explain", "discover", "imagine", "practice",
    "community", "experience", "knowledge", "conversation", "adventure", "freedom",
    "kindness", "patience", "courage", "wisdom", "friendship", "tradition",
]


def _get_json(url: str, timeout: float = 15.0) -> object | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _paragraphs(blocks: list[str]) -> str:
    return "\n\n".join(b.strip() for b in blocks if b.strip()) + "\n"


def fetch_summary_long(title: str, max_sentences: int = 6, timeout: float = 12.0) -> str | None:
    """Wikipedia REST summary with more sentences for prose training."""
    path = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    extract = (data.get("extract") or "").strip()
    if not extract or extract.endswith("may refer to:"):
        return None
    parts = extract.replace("\n", " ").split(". ")
    text = ". ".join(parts[:max_sentences]).strip()
    if text and not text.endswith("."):
        text += "."
    return text


def fetch_wikipedia_prose(
    topics: list[str] | None = None,
    delay: float = 0.35,
) -> str:
    """Wikipedia intro text as plain paragraphs (REST API, rate-limit friendly)."""
    topics = topics or list(dict.fromkeys(DEFAULT_TOPICS + EXTRA_TOPICS))
    paragraphs: list[str] = []

    for title in topics:
        body = fetch_summary_long(title, max_sentences=6)
        if body:
            paragraphs.append(f"{title}. {body}")
        time.sleep(delay)

    if not paragraphs:
        return ""
    return section("Wikipedia Prose", _paragraphs(paragraphs))


def fetch_wikipedia_random_prose(count: int = 50, delay: float = 0.15) -> str:
    """Random Wikipedia summaries as standalone paragraphs."""
    paragraphs: list[str] = []
    for _ in range(count):
        data = _get_json("https://en.wikipedia.org/api/rest_v1/page/random/summary")
        if not isinstance(data, dict):
            time.sleep(delay)
            continue
        title = (data.get("title") or "").strip()
        extract = _clean(data.get("extract") or "")
        if not title or len(extract) < 40 or extract.endswith("may refer to:"):
            time.sleep(delay)
            continue
        paragraphs.append(f"{title}. {extract}")
        time.sleep(delay)

    if not paragraphs:
        return ""
    return section("Random Articles", _paragraphs(paragraphs))


def fetch_nasa_prose(count: int = 20, delay: float = 0.2) -> str:
    """NASA astronomy explanations as plain science writing."""
    url = f"https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY&count={count}"
    data = _get_json(url, timeout=25.0)
    if not isinstance(data, list):
        return ""

    paragraphs: list[str] = []
    for item in data:
        title = (item.get("title") or "").strip()
        explanation = _clean(item.get("explanation") or "")
        if not title or len(explanation) < 40:
            continue
        paragraphs.append(f"{title}. {explanation}")

    if not paragraphs:
        return ""
    return section("Science Writing", _paragraphs(paragraphs))


def fetch_poetry_prose(count: int = 30, max_lines: int = 12, delay: float = 0.12) -> str:
    """Poetry excerpts as plain literary text."""
    blocks: list[str] = []
    for _ in range(count):
        data = _get_json("https://poetrydb.org/random/1")
        if not isinstance(data, list) or not data:
            time.sleep(delay)
            continue
        poem = data[0]
        title = (poem.get("title") or "Untitled").strip()
        author = (poem.get("author") or "Unknown").strip()
        lines = [ln.strip() for ln in (poem.get("lines") or []) if ln.strip()]
        if not lines:
            time.sleep(delay)
            continue
        excerpt = "\n".join(lines[:max_lines])
        blocks.append(f"{title} by {author}\n{excerpt}")
        time.sleep(delay)

    if not blocks:
        return ""
    return section("Poetry", "\n\n".join(blocks) + "\n")


def fetch_dictionary_prose(words: list[str] | None = None, delay: float = 0.1) -> str:
    """Dictionary entries rewritten as plain explanatory sentences."""
    words = words or COMMON_WORDS
    sentences: list[str] = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        data = _get_json(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word)}"
        )
        if not isinstance(data, list) or not data:
            time.sleep(delay)
            continue
        defs: list[str] = []
        for meaning in (data[0].get("meanings") or [])[:2]:
            for definition in (meaning.get("definitions") or [])[:1]:
                text = (definition.get("definition") or "").strip()
                if text:
                    defs.append(text)
        if not defs:
            time.sleep(delay)
            continue
        sentences.append(f"The word \"{word}\" means {defs[0].rstrip('.')}.")
        time.sleep(delay)

    if not sentences:
        return ""
    return section("Word Meanings", _paragraphs(sentences))


def fetch_trivia_prose(count: int = 60, delay: float = 0.18) -> str:
    """Trivia rewritten as plain factual statements."""
    sentences: list[str] = []
    remaining = count
    while remaining > 0:
        batch = min(50, remaining)
        data = _get_json(f"https://opentdb.com/api.php?amount={batch}&type=multiple")
        if not isinstance(data, dict) or not data.get("results"):
            break
        for item in data["results"]:
            question = _clean(re.sub(r"<[^>]+>", "", item.get("question", "")))
            answer = _clean(re.sub(r"<[^>]+>", "", item.get("correct_answer", "")))
            if not question or not answer:
                continue
            q = question.rstrip("?") + "?"
            sentences.append(f"{q} The answer is {answer.rstrip('.')}.")
        remaining -= batch
        time.sleep(delay)

    if not sentences:
        return ""
    return section("General Knowledge", _paragraphs(sentences))


def build_prose_corpus(
    *,
    wiki_topics: list[str] | None = None,
    wiki_random_count: int = 50,
    nasa_count: int = 20,
    poetry_count: int = 30,
    trivia_count: int = 60,
) -> str:
    """Plain English paragraphs from multiple free APIs."""
    parts: list[str] = []
    fetchers = [
        ("wikipedia prose", lambda: fetch_wikipedia_prose(wiki_topics)),
        ("random articles", lambda: fetch_wikipedia_random_prose(wiki_random_count)),
        ("science writing", lambda: fetch_nasa_prose(nasa_count)),
        ("poetry", lambda: fetch_poetry_prose(poetry_count)),
        ("word meanings", lambda: fetch_dictionary_prose()),
        ("general knowledge", lambda: fetch_trivia_prose(trivia_count)),
    ]
    for name, fn in fetchers:
        try:
            block = fn()
            if block:
                parts.append(block)
                print(f"  + {name}: {len(block):,} chars")
        except Exception as exc:
            print(f"  ! {name} skipped: {exc}")

    if not parts:
        raise RuntimeError("No prose content fetched. Check your internet connection.")

    return "\n".join(parts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch plain English prose for training")
    parser.add_argument("--print", action="store_true", help="print corpus to stdout")
    args = parser.parse_args()

    print("Fetching plain prose from web APIs...")
    corpus = build_prose_corpus()
    print(f"Total: {len(corpus):,} chars")
    if args.print:
        print(corpus)


if __name__ == "__main__":
    main()
