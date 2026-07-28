"""Generate curated char-level training data for the mini LLM."""

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "data.txt"


def section(title: str, body: str) -> str:
    return f"\n\n=== {title} ===\n\n{body.strip()}\n"


def build_dialogue() -> str:
    pairs = [
        ("Hello!", "Hello! How can I help you today?"),
        ("Hi there.", "Hi! Nice to meet you. What would you like to know?"),
        ("Good morning.", "Good morning! I hope you have a great day."),
        ("Good evening.", "Good evening! How was your day?"),
        ("How are you?", "I am doing well, thank you for asking."),
        ("What is your name?", "I am a small language model learning to write text."),
        ("Who made you?", "I was built step by step in Python as a learning project."),
        ("Thank you.", "You are welcome! Glad I could help."),
        ("Thanks a lot.", "Any time! Let me know if you need anything else."),
        ("Goodbye.", "Goodbye! Take care and talk to you soon."),
        ("See you later.", "See you later! Have a good one."),
        ("Can you help me?", "Yes, I can try. Tell me what you are working on."),
        ("I need advice.", "Share the details and I will do my best to respond clearly."),
        ("What is Python?", "Python is a programming language that is easy to read and write."),
        ("What is a model?", "A model learns patterns from data and predicts what comes next."),
        ("What is training?", "Training means showing the model many examples so it improves."),
        ("What is a token?", "A token is a piece of text, often a character or word part."),
        ("Explain AI simply.", "AI is software that finds patterns in data to make predictions."),
        ("I feel stuck.", "That is normal when learning. Take one small step at a time."),
        ("This is hard.", "Hard things become easier with practice and patience."),
        ("What should I learn first?", "Start with the basics, then build one small project."),
        ("How do I get better at coding?", "Write code every day, read others code, and fix bugs."),
        ("What is an LLM?", "An LLM is a large language model trained on lots of text."),
        ("What is a neural network?", "It is layers of math that adjust weights to reduce error."),
        ("What is loss?", "Loss measures how wrong the model predictions are."),
        ("What is a checkpoint?", "A saved copy of model weights after training."),
        ("What is a tokenizer?", "It converts text to numbers and back again."),
        ("What is attention?", "Attention lets the model focus on relevant parts of the input."),
        ("Tell me a fun fact.", "Honey never spoils. Archaeologists found edible honey in tombs."),
        ("Tell me another fact.", "Octopuses have three hearts and blue blood."),
        ("What is two plus two?", "Two plus two equals four."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What color is the sky?", "The sky is usually blue on a clear day."),
        ("Recommend a habit.", "Try writing for ten minutes each morning."),
        ("I am tired.", "Rest is important. A short break can help you focus again."),
        ("I am excited.", "That is great! Channel that energy into your next task."),
        ("What is the weather like?", "I cannot see the weather, but I hope it is nice where you are."),
        ("Sing me a line.", "I cannot sing, but I can write: the sun sets slow and soft."),
        ("Write a short poem.", "Soft rain on glass, quiet thoughts pass, code waits, learning lasts."),
        ("Give me a motto.", "Build small, learn daily, improve steadily."),
        ("Motivate me.", "You are closer than yesterday. Keep going."),
        ("What is open source?", "Open source means code is shared for anyone to use and study."),
        ("What is Git?", "Git tracks changes in files and helps teams collaborate on code."),
        ("What is a bug?", "A bug is a mistake in code that causes wrong behavior."),
        ("How do I fix bugs?", "Read the error, reproduce the issue, then change one thing at a time."),
    ]

    lines = []
    for user, assistant in pairs:
        lines.append(f"User: {user}\nAssistant: {assistant}\n")
        # Variation with lowercase user prompt
        if user[0].isupper() and len(user) > 1:
            low = user[0].lower() + user[1:]
            lines.append(f"User: {low}\nAssistant: {assistant}\n")
    return "".join(lines)


def build_code_snippets() -> str:
    snippets = [
        '''# read training data
text = open("data/data.txt", encoding="utf-8").read()
print(len(text))''',
        '''def add(a, b):
    return a + b

result = add(2, 3)
print(result)''',
        '''for i in range(5):
    print(i)''',
        '''if loss < best_loss:
    best_loss = loss
    save_checkpoint(model)''',
        '''class CharTokenizer:
    def encode(self, s):
        return [self.stoi[c] for c in s]''',
        '''import torch
model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)''',
        '''x, y = batch
logits, loss = model(x, y)
loss.backward()
optimizer.step()''',
        '''prompt = "User: Hello!"
tokens = tokenizer.encode(prompt)
output = model.generate(tokens, max_new_tokens=100)''',
    ]
    return "\n\n".join(f"Example code:\n{s}\n" for s in snippets)


def build_prose() -> str:
    paragraphs = [
        "Learning to build a language model starts with simple pieces. First you read text. "
        "Then you turn letters into numbers. Then you train a network to predict the next letter. "
        "Over time, the model picks up spelling, spacing, and common phrases.",

        "Good training data uses clear sentences, normal punctuation, and varied topics. "
        "Small models cannot learn everything at once. They do best when patterns repeat "
        "in consistent formats, like questions followed by answers.",

        "A char level model sees one letter at a time. It needs many examples of common words "
        "like the, and, is, to, and you. Short lines are easier to learn than long walls of text.",

        "When you prompt the model, give it a few words that match the training style. "
        "If you trained on dialogue, start with User: or Assistant:. The model continues "
        "in the same pattern it saw during training.",

        "Patience matters in machine learning. Loss may drop quickly at first, then slow down. "
        "That is normal. Train longer, use more data, or make the model slightly bigger if "
        "you need better results.",
    ]
    return "\n\n".join(paragraphs)


def build_word_and_spelling() -> str:
    # Common English words and letter patterns for char-level learning
    words = [
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
        "how", "its", "may", "new", "now", "old", "see", "way", "who", "boy",
        "did", "let", "put", "say", "she", "too", "use", "hello", "world",
        "python", "model", "train", "learn", "token", "data", "code", "text",
        "user", "assistant", "question", "answer", "help", "thanks", "please",
    ]
    lines = ["Common words:"]
    lines.append(", ".join(words))
    lines.append("")
    lines.append("Word list:")
    for w in words:
        lines.append(w)
    # Alphabet coverage
    lines.append("")
    lines.append("Alphabet: abcdefghijklmnopqrstuvwxyz")
    lines.append("Alphabet: ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    lines.append("Digits: 0123456789")
    lines.append("Punctuation: . , ! ? : ; ' \" - ( )")
    return "\n".join(lines)


def build_lists() -> str:
    return """
Tips for training a small model:
1. Use clean, consistent text.
2. Include dialogue if you want chat-like output.
3. Train until loss stops improving.
4. Save checkpoints often.
5. Test with short prompts first.

Project files:
- config.py sets hyperparameters.
- tokenizer.py maps text to numbers.
- dataset.py creates training batches.
- model.py defines the transformer.
- train.py runs the training loop.
- generate.py writes new text.

Next token prediction:
Given "Hello", predict " " or next letters.
Given "User:", predict " " then a question.
Given "Assistant:", predict a helpful reply.
"""


def expand_dialogue_corpus(base: str, repeats: int = 8) -> str:
    """Repeat dialogue with light separators so char patterns appear often."""
    chunks = base.strip().split("\n\n")
    out = []
    for r in range(repeats):
        out.append(f"\n--- dialogue block {r + 1} ---\n")
        out.extend(chunks)
    return "\n".join(out)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build or append training data")
    parser.add_argument(
        "--append",
        action="store_true",
        help="add new content to existing data.txt instead of overwriting",
    )
    args = parser.parse_args()

    parts = [
        section("Dialogue", expand_dialogue_corpus(build_dialogue(), repeats=3)),
        section("Prose", build_prose()),
        section("Code", build_code_snippets()),
    ]
    new_text = "\n".join(parts)

    OUT.parent.mkdir(parents=True, exist_ok=True)

    if args.append and OUT.exists() and OUT.read_text(encoding="utf-8").strip():
        existing = OUT.read_text(encoding="utf-8")
        text = existing + "\n\n=== Appended training data ===\n\n" + new_text
        mode = "Appended to"
    else:
        base = [
            "Mini LLM training corpus. Modern English, dialogue, code, and prose.\n",
            section("Dialogue", expand_dialogue_corpus(build_dialogue(), repeats=10)),
            section("Prose", build_prose()),
            section("Code", build_code_snippets()),
            section("Vocabulary", build_word_and_spelling()),
            section("Lists", build_lists()),
        ]
        text = "\n".join(base)
        filler = (build_prose() + "\n\n" + build_dialogue()) * 12
        text = text + "\n\n=== Extended practice corpus ===\n\n" + filler
        mode = "Wrote"

    OUT.write_text(text, encoding="utf-8")
    print(f"{mode} {len(text):,} characters to {OUT}")


if __name__ == "__main__":
    main()
