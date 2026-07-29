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


def build_wiki_articles() -> str:
    """Encyclopedia-style prose — short articles, clear sentences for char-level learning."""
    articles = [
        (
            "Earth",
            "Earth is the third planet from the Sun and the only place known to support life. "
            "It is mostly covered by oceans. The atmosphere contains nitrogen and oxygen. "
            "Earth rotates once every 24 hours and orbits the Sun in about 365 days.",
        ),
        (
            "The Moon",
            "The Moon is Earth's only natural satellite. It is about one quarter the size of Earth. "
            "The Moon has no air and no liquid water on its surface. Its gravity is weaker than Earth's, "
            "which is why astronauts could jump higher during the Apollo missions.",
        ),
        (
            "The Sun",
            "The Sun is a star at the center of our solar system. It is a giant ball of hot gas, "
            "mostly hydrogen and helium. The Sun provides light and heat that make life on Earth possible. "
            "Never look directly at the Sun without proper eye protection.",
        ),
        (
            "Water",
            "Water is a simple molecule made of two hydrogen atoms and one oxygen atom. "
            "It exists as a solid (ice), liquid, or gas (steam). Water dissolves many substances, "
            "which is why it is called the universal solvent. All known life needs water.",
        ),
        (
            "Photosynthesis",
            "Photosynthesis is how green plants make food using sunlight. They take in carbon dioxide "
            "from the air and water from the soil. Using light energy, they produce sugar and release "
            "oxygen. This process supports nearly all life on Earth.",
        ),
        (
            "The Human Brain",
            "The human brain controls thought, memory, emotion, and movement. It contains billions "
            "of nerve cells called neurons. The brain uses about 20 percent of the body's energy even "
            "at rest. Learning new skills can strengthen connections between neurons.",
        ),
        (
            "DNA",
            "DNA stores the instructions for building and running a living organism. "
            "It is shaped like a double helix, discovered by Watson and Crick in 1953. "
            "DNA is made of four chemical bases: adenine, thymine, cytosine, and guanine.",
        ),
        (
            "Evolution",
            "Evolution is the change in inherited traits of populations over many generations. "
            "Charles Darwin proposed natural selection: organisms with helpful traits are more likely "
            "to survive and reproduce. Fossils and DNA evidence support evolutionary theory.",
        ),
        (
            "Ancient Rome",
            "Ancient Rome began as a small city on the Tiber River in Italy. It grew into a vast "
            "empire that ruled much of Europe, North Africa, and the Middle East. Romans built roads, "
            "aqueducts, and laws that influenced many modern nations.",
        ),
        (
            "The Renaissance",
            "The Renaissance was a period of renewed interest in art, science, and classical learning "
            "in Europe, roughly from the 14th to the 17th century. Artists like Leonardo da Vinci and "
            "Michelangelo created famous works. Printing spread ideas faster than ever before.",
        ),
        (
            "World War II",
            "World War II lasted from 1939 to 1945 and involved most of the world's nations. "
            "It ended with Allied victory over Axis powers. The war caused enormous loss of life "
            "and led to the founding of the United Nations in 1945.",
        ),
        (
            "The Internet",
            "The Internet is a global network of connected computers. It grew from ARPANET in the "
            "1960s and 1970s. The World Wide Web, invented by Tim Berners-Lee in 1989, made "
            "information easy to browse with links and web pages.",
        ),
        (
            "Python Programming",
            "Python is a high-level programming language created by Guido van Rossum. "
            "It emphasizes readable code and rapid development. Python is widely used for web apps, "
            "data science, automation, and machine learning.",
        ),
        (
            "Machine Learning",
            "Machine learning is a branch of artificial intelligence where computers learn from data "
            "instead of following only fixed rules. Models adjust internal parameters to reduce error. "
            "Common tasks include classification, prediction, and text generation.",
        ),
        (
            "Electricity",
            "Electricity is the flow of electric charge, usually through wires. "
            "It powers lights, computers, and motors. Static electricity builds up when charges "
            "separate; current electricity flows steadily in a circuit.",
        ),
        (
            "Gravity",
            "Gravity is the force that attracts objects with mass toward each other. "
            "On Earth, gravity gives objects weight and keeps the Moon in orbit. "
            "Isaac Newton described gravity mathematically; Einstein explained it as curved spacetime.",
        ),
        (
            "Volcanoes",
            "A volcano is an opening in Earth's crust where molten rock, ash, and gases escape. "
            "Most volcanoes form at plate boundaries. Active volcanoes can erupt with little warning. "
            "Volcanic soil is often very fertile.",
        ),
        (
            "The Amazon Rainforest",
            "The Amazon rainforest spans several South American countries and holds immense biodiversity. "
            "It produces much of the world's oxygen through photosynthesis and stores huge amounts of carbon. "
            "Deforestation threatens wildlife and indigenous communities.",
        ),
        (
            "Dinosaurs",
            "Dinosaurs were reptiles that dominated Earth for over 150 million years. "
            "They ranged from small bird-like species to enormous plant eaters like Argentinosaurus. "
            "Most dinosaurs went extinct about 66 million years ago, likely after an asteroid impact.",
        ),
        (
            "The Periodic Table",
            "The periodic table organizes all known chemical elements by atomic number and properties. "
            "Elements in the same column often behave similarly. Dmitri Mendeleev published an early "
            "version in 1869 and predicted elements not yet discovered.",
        ),
        (
            "Democracy",
            "Democracy is a system of government where power comes from the people, often through voting. "
            "Ancient Athens practiced direct democracy on a small scale. Modern democracies usually use "
            "elected representatives and protect individual rights.",
        ),
        (
            "Climate Change",
            "Climate change refers to long-term shifts in temperature and weather patterns. "
            "Human activities, especially burning fossil fuels, increase greenhouse gases in the atmosphere. "
            "Effects include rising sea levels, stronger storms, and habitat loss.",
        ),
        (
            "The Solar System",
            "The solar system has eight planets orbiting the Sun. Inner planets are rocky; outer giants "
            "are mostly gas and ice. Asteroids and comets also orbit the Sun. Pluto is classified as a dwarf planet.",
        ),
        (
            "Shakespeare",
            "William Shakespeare was an English playwright and poet who lived from 1564 to 1616. "
            "He wrote tragedies like Hamlet and Macbeth, comedies like A Midsummer Night's Dream, "
            "and sonnets still read today. His work shaped the English language.",
        ),
        (
            "The Printing Press",
            "Johannes Gutenberg's printing press, around 1440, made books cheaper and faster to produce. "
            "Ideas spread across Europe more quickly. Literacy increased. The press helped fuel the "
            "Reformation, science, and the Enlightenment.",
        ),
    ]

    blocks = []
    for title, body in articles:
        blocks.append(f"Article: {title}\n{body}")
    return "\n\n".join(blocks)


def build_wiki_dialogue() -> str:
    """Q&A pairs drawn from encyclopedia topics — matches chat training format."""
    pairs = [
        ("What is Earth?", "Earth is the third planet from the Sun and the only known home of life."),
        ("Tell me about the Moon.", "The Moon orbits Earth and has no air or liquid water on its surface."),
        ("What does the Sun do?", "The Sun is a star that gives Earth light, heat, and energy."),
        ("Why is water important?", "Water is essential for all known life and exists as ice, liquid, and vapor."),
        ("What is photosynthesis?", "Photosynthesis is how plants use sunlight to make food and release oxygen."),
        ("How does the brain work?", "The brain uses billions of neurons to control thought, memory, and movement."),
        ("What is DNA?", "DNA is a molecule that stores genetic instructions for living things."),
        ("Who was Charles Darwin?", "Charles Darwin was a scientist who explained evolution by natural selection."),
        ("What was Ancient Rome?", "Ancient Rome was a powerful civilization that built roads, laws, and aqueducts."),
        ("What was the Renaissance?", "The Renaissance was a European period of renewed art, science, and learning."),
        ("When was World War II?", "World War II lasted from 1939 to 1945."),
        ("What is the Internet?", "The Internet is a global network that connects computers and shares information."),
        ("What is Python used for?", "Python is used for web development, data science, scripting, and machine learning."),
        ("What is machine learning?", "Machine learning lets computers improve by finding patterns in data."),
        ("What is gravity?", "Gravity is the force that pulls objects with mass toward each other."),
        ("What is a volcano?", "A volcano is an opening where molten rock and gases escape from Earth's crust."),
        ("Where is the Amazon?", "The Amazon rainforest is in South America and holds huge biodiversity."),
        ("What happened to the dinosaurs?", "Most dinosaurs went extinct about 66 million years ago."),
        ("What is the periodic table?", "The periodic table lists chemical elements ordered by atomic number."),
        ("What is democracy?", "Democracy is government by the people, often through free elections."),
        ("What is climate change?", "Climate change is long-term warming and shifting weather, partly from human activity."),
        ("How many planets are there?", "There are eight planets in our solar system."),
        ("Who was Shakespeare?", "William Shakespeare was an English playwright who wrote Hamlet and many other works."),
        ("What did Gutenberg invent?", "Gutenberg invented a printing press that spread books and ideas across Europe."),
        ("What is electricity?", "Electricity is the flow of electric charge used to power devices."),
        ("Explain evolution simply.", "Evolution means species change over time through natural selection."),
        ("What is an LLM?", "An LLM is a language model trained on text to predict and generate words."),
        ("What is a transformer?", "A transformer is a neural network architecture that uses attention layers."),
        ("What is open source software?", "Open source software has public code that anyone can study and modify."),
        ("What is a checkpoint in ML?", "A checkpoint saves model weights during training so you can resume later."),
    ]

    lines = []
    for user, assistant in pairs:
        lines.append(f"User: {user}\nAssistant: {assistant}\n")
        if user[0].isupper():
            low = user[0].lower() + user[1:]
            lines.append(f"User: {low}\nAssistant: {assistant}\n")
    return "".join(lines)


def build_wiki_corpus() -> str:
    return (
        section("Wiki Articles", build_wiki_articles())
        + section("Wiki Q&A", build_wiki_dialogue())
    )


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
