"""Natural conversation, casual chat, and story exchanges for training."""

from scripts.build_training_data import section


def _format_conversation(turns: list[tuple[str, str]]) -> str:
    lines = [f"User: {u}\nAssistant: {a}" for u, a in turns]
    return "\n".join(lines) + "\n"


def _qa(pairs: list[tuple[str, str]]) -> str:
    lines = []
    for user, assistant in pairs:
        lines.append(f"User: {user}\nAssistant: {assistant}\n")
        low = user[0].lower() + user[1:] if user and user[0].isupper() else user
        if low != user:
            lines.append(f"User: {low}\nAssistant: {assistant}\n")
    return "".join(lines)


def build_story_exchanges() -> str:
    """Directly teaches: ask for story -> get a story."""
    pairs = [
        (
            "Tell me a story.",
            "Once, a baker opened shop at dawn. Bread warmed the street. A child traded a drawing for a roll. The baker hung it on the wall. Small kindness fed the town for years.",
        ),
        (
            "Tell me a short story.",
            "A lighthouse keeper saw a ship off course in fog. She flashed the lamp twice. The ship corrected course and passed safely. No one on land ever knew.",
        ),
        (
            "Can you tell me a story?",
            "Two friends found a map in an old book. The map led to a garden behind the library. They planted tomatoes there every spring after.",
        ),
        (
            "Story time please.",
            "There was a dog who rode the bus every morning with the same commuter. When the commuter moved away, the driver saved a seat anyway. Habits can be love.",
        ),
        (
            "Give me a story about friendship.",
            "Maya and Jin built a treehouse with mismatched nails and loud laughter. It leaned, but it held them through every summer argument and apology.",
        ),
        (
            "Tell me a bedtime story.",
            "The moon watched a quiet town sleep. A cat crossed roofs without sound. A light went out in a window. The night kept everyone's secrets.",
        ),
        (
            "I want to hear a story.",
            "An old piano sat in a closed school. A student returned years later and played one chord. Dust danced in the light like it remembered.",
        ),
        (
            "Make up a story.",
            "On a rainy bus, strangers shared one umbrella at the stop. None of them spoke the same language. They laughed anyway.",
        ),
    ]
    return _qa(pairs)


def build_casual_chat() -> str:
    conversations = [
        [
            ("hey", "Hey! What's on your mind?"),
            ("not much, you?", "Same here. Happy to chat about anything."),
            ("cool", "Cool. Ask me something whenever you like."),
        ],
        [
            ("hi there", "Hi there! How's your day going?"),
            ("pretty good honestly", "Glad to hear that. Anything fun happen?"),
            ("just coffee and work", "Solid day then. Coffee helps."),
        ],
        [
            ("hello", "Hello! Nice to meet you."),
            ("nice to meet you too", "What would you like to talk about?"),
            ("anything really", "We could do a fun fact, a story, or just chat."),
        ],
        [
            ("how are you?", "I'm doing well, thanks for asking. How are you?"),
            ("I'm okay, bit tired", "Tired days happen. Rest when you can."),
            ("yeah I will", "Good plan."),
        ],
        [
            ("what's up", "Not much. Here to chat if you need anything."),
            ("just bored", "Boredom is a good time to learn something random."),
            ("like what", "Pick a topic: space, history, food, or a short story."),
        ],
        [
            ("thanks for the help", "You're welcome! Glad I could help."),
            ("seriously appreciate it", "Any time. Come back if you need more."),
        ],
        [
            ("sorry if that was a dumb question", "No question is dumb. Ask freely."),
            ("ok good", "That's what I'm here for."),
        ],
        [
            ("wait, can you explain that simpler?", "Sure. I'll use shorter sentences and plain words."),
            ("better, thanks", "Happy to slow down."),
        ],
        [
            ("hmm I'm not sure what to ask", "That's fine. Try: tell me a fun fact, or tell me a story."),
            ("tell me a fun fact", "Honey never spoils. Archaeologists found edible honey in ancient tombs."),
            ("whoa", "Nature is full of surprises."),
        ],
        [
            ("lol that was random", "Fair point. Want something more on topic?"),
            ("yeah tell me about space", "Space is mostly empty, but stars and planets fill it with light and motion."),
        ],
        [
            ("good morning", "Good morning! Hope you have a good day."),
            ("you too", "Thank you!"),
        ],
        [
            ("good night", "Good night! Sleep well."),
            ("night", "Rest easy."),
        ],
        [
            ("I'm confused", "That's okay. Tell me what part is unclear."),
            ("all of it honestly", "Let's start with one small piece at a time."),
        ],
        [
            ("can we start over?", "Of course. What would you like to talk about now?"),
            ("stories", "I can tell you a short story. Just say tell me a story."),
        ],
        [
            ("you're helpful", "Thank you! I try my best."),
            ("keep it up", "Will do."),
        ],
    ]
    return "\n\n".join(_format_conversation(c) for c in conversations)


def build_natural_multi_turn() -> str:
    conversations = [
        [
            ("tell me a story", "There was a fox who crossed the same bridge every dusk. One evening the bridge was gone for repairs. The fox sat, waited, and met the builder. They became friends."),
            ("that was nice", "Glad you liked it. Want another?"),
            ("maybe later", "Any time."),
        ],
        [
            ("what can you do?", "I can chat, share facts, explain ideas, or tell short stories."),
            ("tell me a story then", "A sailor kept a notebook of small good things. Years later it was thicker than any novel. Joy had been counted daily."),
            ("better than I expected", "Stories are my favorite too."),
        ],
        [
            ("random question", "Go for it."),
            ("why is the sky blue?", "Sunlight scatters in the air. Blue light scatters more, so the sky looks blue."),
            ("huh neat", "Physics is everywhere."),
        ],
        [
            ("I feel bored", "Boredom happens. A short story might help pass the time."),
            ("ok one story", "An ant carried a leaf ten times its size. Other ants cleared the path. Teamwork moved mountains, inch by inch."),
            ("cute", "Small stories can reset the mood."),
        ],
        [
            ("wow", "Yeah? What caught your attention?"),
            ("that fact about the sun", "The sun's core is extremely hot, millions of degrees. The surface is cooler but still blazing."),
            ("space is wild", "It really is."),
        ],
        [
            ("okay", "Okay. What next?"),
            ("idk you pick", "How about a quick joke or a fun fact?"),
            ("joke", "Why did the bicycle fall over? It was two tired."),
        ],
    ]
    return "\n\n".join(_format_conversation(c) for c in conversations)


def build_modern_prose() -> str:
    paragraphs = [
        "People text fast and think out loud. Um, well, maybe we should try again. Yeah, that sounds better. Natural speech is messy, and that is normal.",
        "Good conversations wander a little. You ask something, get an answer, react, then ask again. The thread does not have to be perfect to feel human.",
        "When someone says tell me a story, they usually want characters, a place, and something that changes. Not a random fact from another topic.",
        "Modern English mixes short replies with longer ones. Okay. Sure. I see what you mean. Let me explain that in plain language.",
        "Stories can be tiny. One person, one choice, one moment. The point is movement: something starts, something shifts, something ends.",
    ]
    return "\n\n".join(paragraphs)


def _expand(text: str, blocks: int, label: str) -> str:
    return "\n\n".join(f"--- {label} {i + 1} ---\n{text.strip()}" for i in range(blocks))


def build_natural_corpus() -> str:
    stories = _expand(build_story_exchanges(), 5, "story-qa")
    casual = _expand(build_casual_chat(), 4, "casual")
    multi = _expand(build_natural_multi_turn(), 6, "natural-multi")
    prose = _expand(build_modern_prose(), 4, "prose")
    return (
        section("Story Exchanges", stories)
        + section("Casual Chat", casual)
        + section("Natural Multi-turn", multi)
        + section("Modern Prose", prose)
    )
