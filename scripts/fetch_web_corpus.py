"""Fetch modern English text from free public APIs (no API keys)."""

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

USER_AGENT = "MiniLLM-TrainingBot/1.0 (local learning project)"
COMMON_WORDS = [
    "hello", "world", "learn", "story", "friend", "music", "ocean", "forest",
    "happy", "quiet", "bright", "travel", "science", "history", "language",
    "memory", "dream", "garden", "winter", "summer", "bridge", "planet", "energy",
    "culture", "family", "health", "school", "market", "village", "mountain",
    "river", "cloud", "light", "shadow", "voice", "truth", "change", "future",
    "curious", "honest", "patient", "creative", "simple", "modern", "ancient",
]


def _get_json(url: str, timeout: float = 12.0) -> object | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def _get_text(url: str, timeout: float = 12.0) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8").strip()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def _clean_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def _qa(user: str, assistant: str) -> str:
    return f"User: {user}\nAssistant: {assistant}\n"


def fetch_trivia(count: int = 80, delay: float = 0.2) -> str:
    """Open Trivia DB — modern quiz Q&A in English."""
    blocks: list[str] = []
    remaining = count
    while remaining > 0:
        batch = min(50, remaining)
        url = f"https://opentdb.com/api.php?amount={batch}&type=multiple"
        data = _get_json(url)
        if not isinstance(data, dict) or not data.get("results"):
            break
        for item in data["results"]:
            question = _clean_html(item.get("question", ""))
            answer = _clean_html(item.get("correct_answer", ""))
            if not question or not answer:
                continue
            blocks.append(_qa(question, answer))
            blocks.append(_qa(f"Quick quiz: {question}", f"The answer is {answer}."))
        remaining -= batch
        time.sleep(delay)

    if not blocks:
        return ""
    return section("Trivia Q&A", "".join(blocks))


def fetch_dictionary(words: list[str] | None = None, delay: float = 0.12) -> str:
    """Free Dictionary API — clear modern definitions."""
    words = words or COMMON_WORDS
    lines: list[str] = []
    qa: list[str] = []
    for word in words:
        data = _get_json(f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word)}")
        if not isinstance(data, list) or not data:
            time.sleep(delay)
            continue
        entry = data[0]
        meanings = entry.get("meanings") or []
        defs: list[str] = []
        for meaning in meanings[:2]:
            for definition in (meaning.get("definitions") or [])[:2]:
                text = (definition.get("definition") or "").strip()
                if text:
                    defs.append(text)
        if not defs:
            time.sleep(delay)
            continue
        summary = defs[0]
        if len(defs) > 1:
            summary += " " + defs[1]
        lines.append(f"Word: {word}\nMeaning: {summary}\n")
        qa.append(_qa(f"What does {word} mean?", summary))
        qa.append(_qa(f"Define {word}.", summary))
        time.sleep(delay)

    if not lines:
        return ""
    return section("Dictionary", "\n".join(lines) + "\n" + "".join(qa))


def fetch_poetry(count: int = 25, max_lines: int = 8, delay: float = 0.15) -> str:
    """PoetryDB — literary English excerpts."""
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
        excerpt = " ".join(lines[:max_lines])
        if len(excerpt) > 420:
            excerpt = excerpt[:417].rsplit(" ", 1)[0] + "..."
        blocks.append(f"Poem: {title} by {author}\n{excerpt}\n")
        blocks.append(
            _qa(
                f"Share a poem by {author}.",
                f"Here is an excerpt from {title}: {excerpt}",
            )
        )
        time.sleep(delay)

    if not blocks:
        return ""
    return section("Poetry", "\n".join(blocks))


def fetch_wikipedia_random(count: int = 40, delay: float = 0.18) -> str:
    """Wikipedia REST API — random article summaries."""
    articles: list[str] = []
    qa: list[str] = []
    for _ in range(count):
        data = _get_json("https://en.wikipedia.org/api/rest_v1/page/random/summary")
        if not isinstance(data, dict):
            time.sleep(delay)
            continue
        title = (data.get("title") or "").strip()
        extract = (data.get("extract") or "").strip()
        if not title or not extract or extract.endswith("may refer to:"):
            time.sleep(delay)
            continue
        parts = extract.replace("\n", " ").split(". ")
        summary = ". ".join(parts[:3]).strip()
        if summary and not summary.endswith("."):
            summary += "."
        articles.append(f"Article: {title}\n{summary}\n")
        qa.append(_qa(f"Tell me about {title}.", summary))
        qa.append(_qa(f"What is {title}?", summary))
        time.sleep(delay)

    if not articles:
        return ""
    return section("Wikipedia Random", "\n".join(articles) + "\n" + "".join(qa))


def fetch_nasa_apod(count: int = 15, delay: float = 0.25) -> str:
    """NASA APOD — science writing in plain English."""
    url = f"https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY&count={count}"
    data = _get_json(url, timeout=20.0)
    if not isinstance(data, list):
        return ""

    blocks: list[str] = []
    for item in data:
        title = (item.get("title") or "").strip()
        explanation = (item.get("explanation") or "").replace("\n", " ").strip()
        if not title or not explanation:
            continue
        parts = explanation.split(". ")
        summary = ". ".join(parts[:3]).strip()
        if summary and not summary.endswith("."):
            summary += "."
        blocks.append(f"Space: {title}\n{summary}\n")
        blocks.append(_qa(f"Tell me about {title}.", summary))
        blocks.append(_qa("Something cool about space?", f"{title}: {summary}"))

    if not blocks:
        return ""
    return section("NASA Space Facts", "\n".join(blocks))


def fetch_numbers_facts(count: int = 20, delay: float = 0.3) -> str:
    """Numbers API — short trivia sentences (best-effort; can be slow)."""
    blocks: list[str] = []
    tried = 0
    while len(blocks) < count and tried < count * 3:
        n = random.randint(1, 9999)
        fact = _get_text(f"http://numbersapi.com/{n}/trivia", timeout=8.0)
        tried += 1
        if not fact or len(fact) < 10:
            time.sleep(delay)
            continue
        blocks.append(f"Fact: {fact}\n")
        blocks.append(_qa("Tell me a random number fact.", fact))
        time.sleep(delay)

    if not blocks:
        return ""
    return section("Number Facts", "\n".join(blocks))


def build_web_corpus(
    *,
    trivia_count: int = 80,
    dictionary_words: list[str] | None = None,
    poetry_count: int = 25,
    wiki_random_count: int = 40,
    nasa_count: int = 15,
    numbers_count: int = 12,
) -> str:
    """Combine all working free-API sources into one training block."""
    parts: list[str] = []
    fetchers = [
        ("trivia", lambda: fetch_trivia(trivia_count)),
        ("dictionary", lambda: fetch_dictionary(dictionary_words)),
        ("poetry", lambda: fetch_poetry(poetry_count)),
        ("wikipedia random", lambda: fetch_wikipedia_random(wiki_random_count)),
        ("nasa", lambda: fetch_nasa_apod(nasa_count)),
        ("numbers", lambda: fetch_numbers_facts(numbers_count)),
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
        raise RuntimeError("No web content fetched. Check your internet connection.")

    return "\n".join(parts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch modern English from free web APIs")
    parser.add_argument("--print", action="store_true", help="print corpus to stdout")
    args = parser.parse_args()

    print("Fetching from free APIs...")
    corpus = build_web_corpus()
    print(f"Total: {len(corpus):,} chars")
    if args.print:
        print(corpus)


if __name__ == "__main__":
    main()
