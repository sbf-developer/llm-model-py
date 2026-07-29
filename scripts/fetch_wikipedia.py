"""Fetch plain-English Wikipedia summaries for training data."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from scripts.build_training_data import section

# diverse topics: history, science, places, culture, tech
DEFAULT_TOPICS = [
    "Human", "Earth", "Sun", "Moon", "Water", "Evolution", "DNA", "Brain",
    "Python (programming language)", "Internet", "World War II", "Ancient Rome",
    "Shakespeare", "Music", "Art", "Democracy", "Climate change", "Photosynthesis",
    "Mathematics", "Physics", "Chemistry", "Biology", "Philosophy", "Psychology",
    "Japan", "India", "France", "United States", "Amazon rainforest", "Mount Everest",
    "Black hole", "Solar System", "Electricity", "Computer", "Artificial intelligence",
    "Novel", "Poetry", "Film", "Olympic Games", "Medicine", "Education", "Language",
    "Economics", "Architecture", "Food", "Coffee", "Dog", "Cat", "Ocean",
]


def fetch_summary(title: str, timeout: float = 10.0) -> str | None:
    path = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{path}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MiniLLM-TrainingBot/1.0 (local learning project)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    extract = data.get("extract", "").strip()
    if not extract or extract.endswith("may refer to:"):
        return None
    # keep first ~2 sentences for char-level model
    parts = extract.replace("\n", " ").split(". ")
    text = ". ".join(parts[:3]).strip()
    if text and not text.endswith("."):
        text += "."
    return text


def build_wikipedia_corpus(
    topics: list[str] | None = None,
    delay: float = 0.15,
) -> str:
    topics = topics or DEFAULT_TOPICS
    articles = []
    for title in topics:
        body = fetch_summary(title)
        if body:
            articles.append(f"Article: {title}\n{body}")
        time.sleep(delay)

    if not articles:
        raise RuntimeError("No Wikipedia articles fetched. Check your internet connection.")

    qa_lines = []
    for block in articles:
        title = block.split("\n", 1)[0].replace("Article: ", "")
        summary = block.split("\n", 1)[1]
        qa_lines.append(f"User: Tell me about {title}.\nAssistant: {summary}\n")
        qa_lines.append(f"User: What is {title}?\nAssistant: {summary}\n")

    body = "\n\n".join(articles) + "\n\n" + "".join(qa_lines)
    return section("Wikipedia", body)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Wikipedia summaries for training")
    parser.add_argument("--topics", nargs="+", help="article titles to fetch")
    parser.add_argument("--print", action="store_true", help="print corpus to stdout")
    args = parser.parse_args()

    corpus = build_wikipedia_corpus(args.topics)
    if args.print:
        print(corpus)
    else:
        print(f"Fetched {corpus.count('Article:')} articles, {len(corpus):,} chars")


if __name__ == "__main__":
    main()
