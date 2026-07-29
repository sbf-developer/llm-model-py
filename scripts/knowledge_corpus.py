"""Broad human-knowledge training data: wiki, science, psychology, stories, jokes, life."""

from scripts.build_training_data import section


def _format_conversation(turns: list[tuple[str, str]]) -> str:
    lines = [f"User: {u}\nAssistant: {a}" for u, a in turns]
    return "\n".join(lines) + "\n"


def _articles(pairs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"Article: {t}\n{b}" for t, b in pairs)


def _qa(pairs: list[tuple[str, str]]) -> str:
    lines = []
    for user, assistant in pairs:
        lines.append(f"User: {user}\nAssistant: {assistant}\n")
        if user and user[0].isupper():
            low = user[0].lower() + user[1:]
            lines.append(f"User: {low}\nAssistant: {assistant}\n")
    return "".join(lines)


def _expand(text: str, blocks: int, label: str) -> str:
    return "\n\n".join(f"--- {label} {i + 1} ---\n{text.strip()}" for i in range(blocks))


def build_general_wiki() -> str:
    return _articles([
        ("Ancient Egypt", "Ancient Egypt flourished along the Nile River for thousands of years. Pharaohs ruled as kings and gods. Egyptians built pyramids, developed writing called hieroglyphs, and mummified the dead."),
        ("The Roman Empire", "Rome grew from a city-state into an empire spanning three continents. Roman law, roads, and Latin language shaped Europe. The empire split and eventually fell in the west in 476 CE."),
        ("The Middle Ages", "Medieval Europe spanned roughly 500 to 1500 CE. Feudalism organized society. Castles, knights, and monasteries were common. The plague and wars reshaped populations."),
        ("Industrial Revolution", "Starting in the 1700s, machines and factories transformed production. Steam power, textiles, and railways changed cities. Living standards rose over time but working conditions were often harsh."),
        ("World War I", "World War I lasted from 1914 to 1918. Trench warfare devastated Europe. New technology included tanks, planes, and poison gas. The war redrew maps and planted seeds for World War II."),
        ("The Cold War", "After 1945, the United States and Soviet Union competed without direct war. Nuclear weapons, space races, and proxy conflicts defined the era. The Soviet Union dissolved in 1991."),
        ("Japan", "Japan is an island nation in East Asia. It blends ancient traditions with modern technology. Tokyo is among the largest cities in the world. Japan exports cars, electronics, and culture."),
        ("India", "India has over a billion people and deep history. It gave the world decimal numbers, yoga, and diverse religions. Modern India is a democracy and growing economy."),
        ("The Amazon", "The Amazon River basin holds the world's largest rainforest. It supports incredible biodiversity. Indigenous peoples have lived there for millennia. Conservation protects global climate."),
        ("Antarctica", "Antarctica is the coldest continent, covered in ice. No country owns it; science treaties govern use. Penguins and seals live there. Ice cores record ancient climate."),
        ("Music", "Music uses rhythm, melody, and harmony to express emotion. Instruments and voices create sound. Genres include classical, jazz, rock, hip-hop, and folk. Every culture has music."),
        ("Art", "Art includes painting, sculpture, photography, and more. It communicates ideas and beauty. Famous movements include Renaissance, impressionism, and modern art."),
        ("Language", "Language lets humans share complex ideas. Thousands of languages exist worldwide. Writing preserves speech across time. Translation connects cultures."),
        ("Food and Cooking", "Cooking transforms raw ingredients with heat and skill. Recipes pass through families and cultures. Balanced meals include protein, vegetables, and grains. Food safety prevents illness."),
        ("Sports", "Sports combine physical skill, rules, and competition. Team sports build cooperation. Exercise improves health. The Olympics celebrate global athletic achievement."),
        ("Money and Trade", "Money is a medium of exchange. Trade lets people specialize and swap goods. Banks store and lend money. Inflation means prices rise over time."),
        ("Law and Justice", "Laws are rules societies enforce. Courts resolve disputes. Rights protect individuals from abuse. Democracies aim for fair trials and equal treatment."),
        ("Medicine", "Medicine prevents and treats disease. Doctors study the body and evidence. Vaccines train immunity. Public health protects communities."),
        ("Architecture", "Architecture designs buildings for shelter and beauty. Styles reflect culture and materials. Safety codes prevent collapse and fire. Cities shape daily life."),
        ("Transportation", "Humans move by foot, wheel, rail, air, and sea. Roads and ports connect economies. Public transit reduces congestion. Safety rules save lives."),
        ("Ancient Greece", "Ancient Greece developed democracy, philosophy, and theater. City-states like Athens and Sparta competed. Greek ideas still shape science and politics."),
        ("The Renaissance", "The Renaissance revived classical learning in Europe from the 1400s onward. Art, science, and exploration expanded. Printing spread knowledge faster than ever."),
        ("The United States", "The United States declared independence in 1776. It grew across North America. The country mixes many immigrant cultures and a federal government."),
        ("China", "China has ancient civilization and modern industry. The Great Wall and silk road mark its history. Today it is a major economic power."),
        ("Africa", "Africa is a vast continent with deserts, savannas, and rainforests. It is the cradle of humankind. Diverse languages and nations thrive there."),
        ("The Olympics", "The modern Olympics began in 1896. Athletes compete in summer and winter sports. The games promote peace and excellence across nations."),
        ("The Moon Landing", "In 1969, Apollo 11 landed humans on the Moon. Neil Armstrong and Buzz Aldrin walked the surface. The mission showed what cooperation can achieve."),
        ("The Internet Age", "The internet connects billions of devices. Email, search, and social media changed daily life. Privacy and truth become important civic questions."),
        ("Climate and Weather", "Weather is short-term atmosphere; climate is long-term patterns. Greenhouse gases trap heat. Storms, droughts, and seas respond to warming."),
        ("Oceans", "Oceans cover most of Earth. They regulate climate and host marine life. Coral reefs support rich ecosystems. Plastic pollution threatens wildlife."),
        ("Mountains", "Mountains form when plates collide or volcanoes rise. They affect weather and water supply. Climbing requires skill and respect for nature."),
        ("Deserts", "Deserts receive very little rain. Plants and animals adapt to dryness. Sahara and Gobi are famous examples. Deserts are not lifeless."),
        ("Cities", "Cities cluster people, jobs, and culture. They need housing, transit, and services. Urban planning shapes quality of life."),
        ("Farming", "Agriculture feeds populations. Crops and livestock need soil, water, and care. Sustainable farming protects land for future harvests."),
        ("Books and Libraries", "Books store knowledge and stories. Libraries lend books freely to communities. Reading builds vocabulary and empathy."),
        ("Film and Television", "Moving images tell stories at scale. Documentaries teach; fiction explores emotion. Media shapes how societies see themselves."),
        ("Human Rights", "Human rights are basic freedoms all people deserve. They include speech, safety, and fair treatment. Movements expand rights over time."),
        ("Volunteering", "Volunteering gives time to help others without pay. Food banks, tutoring, and cleanups need helpers. Service builds community."),
        ("Invention", "Invention solves problems with new tools or methods. Wheels, vaccines, and smartphones changed life. Curiosity and trial drive invention."),
    ])


def build_psychology() -> str:
    return _articles([
        ("Emotions", "Emotions like joy, fear, anger, and sadness guide behavior. They signal what matters to us. Naming feelings helps manage them. All emotions are normal in context."),
        ("Stress", "Stress is the body's response to pressure. Short stress can sharpen focus. Chronic stress harms sleep and health. Rest, exercise, and support reduce stress."),
        ("Memory", "Memory stores experiences and knowledge. Short-term memory holds brief information. Long-term memory can last years. Sleep helps consolidate memories."),
        ("Learning", "Learning changes the brain through practice and feedback. Mistakes are part of growth. Spaced repetition strengthens recall. Curiosity fuels motivation."),
        ("Habits", "Habits are automatic behaviors triggered by cues. Small consistent actions build change. Environment shapes habits. Replacing a bad habit works better than only stopping it."),
        ("Sleep", "Sleep restores body and mind. Most adults need seven to nine hours. Screens and caffeine can disrupt sleep. Regular schedules help quality."),
        ("Motivation", "Motivation is the drive to act toward goals. Intrinsic motivation comes from interest. Extrinsic motivation uses rewards. Clear small steps sustain effort."),
        ("Empathy", "Empathy is understanding another person's feelings. Listening without judging builds connection. Empathy differs from agreeing with everything."),
        ("Confidence", "Confidence is trust in your ability to try. It grows with practice, not perfection. Self-talk shapes confidence. Setbacks are data, not identity."),
        ("Grief", "Grief follows loss of people, jobs, or dreams. It has no fixed timeline. Support from others helps. Feeling grief means something mattered."),
        ("Anxiety", "Anxiety is worry about future threats. Mild anxiety can prepare us. Persistent anxiety may need coping skills or professional help. Breathing slowly can calm the body."),
        ("Relationships", "Healthy relationships need respect, honesty, and boundaries. Communication prevents small issues becoming big ones. Trust builds slowly and breaks quickly."),
        ("Attention", "Attention selects what the mind processes. Multitasking often reduces quality. Breaks restore focus. Deep work needs fewer distractions."),
        ("Personality", "Personality is stable patterns of thinking and behaving. Traits vary across people. Context influences behavior. People can grow while keeping core traits."),
        ("Child Development", "Children learn language, motor skills, and social rules over years. Play is serious learning. Safe attachment supports exploration. Patience matters."),
        ("Depression", "Depression involves persistent low mood and loss of interest. It is a real medical condition, not laziness. Talk therapy, lifestyle, and medicine can help."),
        ("Self-Esteem", "Self-esteem is how you value yourself. It shifts with experience. Kind self-talk and achievable goals support healthy esteem."),
        ("Burnout", "Burnout comes from long-term stress at work or care roles. Exhaustion, cynicism, and reduced performance are signs. Rest and boundaries are essential."),
        ("Forgiveness", "Forgiveness releases resentment; it does not require forgetting harm. It can reduce personal suffering. Safety comes first in abusive situations."),
        ("Gratitude", "Gratitude notices good things deliberately. It can improve mood over time. A simple daily list of three items helps many people."),
        ("Mindfulness", "Mindfulness observes thoughts and sensations without harsh judgment. It trains attention. Short daily practice builds the skill."),
        ("Cognitive Bias", "Biases are mental shortcuts that sometimes mislead. Confirmation bias favors what we already believe. Awareness reduces some errors."),
        ("Body Language", "Posture, gaze, and gestures communicate mood. Open posture can signal friendliness. Context matters across cultures."),
        ("Resilience", "Resilience recovers from adversity. Support, meaning, and flexible thinking strengthen it. Resilience can be practiced."),
        ("Procrastination", "Procrastination delays important tasks. Fear of failure and unclear steps often cause it. Start with five minutes to break inertia."),
        ("Active Listening", "Active listening gives full attention and reflects back what you heard. It reduces misunderstandings. Questions show care."),
        ("Anger", "Anger signals perceived injustice or threat. Express it calmly when safe. Physical activity can discharge intense arousal."),
        ("Shame vs Guilt", "Guilt says I did something bad. Shame says I am bad. Guilt can motivate repair; shame often paralyzes."),
        ("Flow State", "Flow is deep focus where time feels altered. It happens when skill matches challenge. Clear goals and few interruptions help."),
    ])


def build_human_experience() -> str:
    return _articles([
        ("First Day at Work", "Starting a new job brings nerves and excitement. Ask questions early. Listen more than you talk at first. Small wins build reputation."),
        ("Making Friends", "Friendship grows from shared time and trust. Be curious about others. Show up when you say you will. Quality beats quantity."),
        ("Dealing with Failure", "Failure means an attempt did not reach the goal. It provides feedback. Many successes follow many failures. Persistence and adjustment matter."),
        ("Moving to a New Place", "Relocation disrupts routines. Explore gradually. Meet neighbors and local spots. Homesickness fades as new roots grow."),
        ("Learning a Skill", "Skills improve with deliberate practice. Break tasks into steps. Get feedback from someone better. Celebrate small progress."),
        ("Handling Conflict", "Conflict is disagreement, not always enemy. Stay calm and use I statements. Seek understanding before winning. Compromise can help both sides."),
        ("Being Tired", "Fatigue slows thinking and mood. Rest is productive, not lazy. Food, water, and movement affect energy. Chronic tiredness deserves attention."),
        ("Celebrating Success", "Celebrating reinforces effort. Share credit with helpers. Rest before the next challenge. Success is a moment, not the whole story."),
        ("Asking for Help", "Asking for help is strength, not weakness. People often want to assist. Be specific about what you need. Offer help in return when you can."),
        ("Loneliness", "Loneliness is feeling disconnected. It is common and painful. Reach out to one person. Groups around hobbies can help."),
        ("Parenting Basics", "Parenting is long-term care and teaching. Consistency and love matter. No one is perfect. Children need safety and autonomy as they grow."),
        ("Aging", "Aging changes body and priorities over time. Experience brings perspective. Stay active mentally and physically. Relationships deepen with years."),
        ("Travel", "Travel exposes new food, language, and customs. Planning reduces stress. Respect local norms. Memories often outlast souvenirs."),
        ("Creativity", "Creativity combines old ideas in new ways. Constraints can spark invention. Start messy, refine later. Everyone can create something."),
        ("Patience", "Patience tolerates delay without rage. Big goals take time. Impatience signals caring. Breathe and focus on the next step."),
        ("Starting Over", "Starting over follows job loss, breakups, or mistakes. Identity is more than one chapter. Support and small routines anchor you."),
        ("Public Speaking", "Speaking to groups scares many people. Prepare key points, not every word. Practice aloud and breathe before you begin."),
        ("Saving Money", "Saving builds a buffer for emergencies. Pay yourself first with a small automatic transfer. Needs come before wants."),
        ("Cooking at Home", "Home cooking saves money and controls ingredients. Learn one dish well, then expand. Mistakes are edible lessons."),
        ("Exercise Routines", "Regular movement strengthens heart and mood. Walking counts. Consistency beats extreme workouts that burn out."),
        ("Breakups", "Breakups hurt even when necessary. Grief and relief can mix. Time, friends, and new routines help healing."),
        ("Job Interviews", "Interviews test fit and skills. Research the role and prepare examples. Ask thoughtful questions at the end."),
        ("Roommates", "Shared housing needs clear rules about bills, noise, and cleaning. Talk early about problems. Respect private space."),
        ("Pet Care", "Pets need food, water, exercise, and vet care. Adoption is a long commitment. Animals return affection and routine."),
        ("Neighborhood Life", "Neighbors share streets and sometimes tools or watchful eyes. Small kindnesses build trust. Introduce yourself when you move in."),
        ("Holiday Stress", "Holidays mix joy and pressure. Budget time and money. It is fine to simplify traditions."),
        ("Learning Online", "Online courses offer flexible learning. Schedule fixed study times. Take notes actively, not passively."),
        ("Apologizing", "A good apology names the harm without excuses. Changed behavior matters more than repeated sorry."),
        ("Saying No", "Saying no protects time and health. Brief and kind refusals beat overcommitting. You cannot please everyone."),
    ])


def build_stories() -> str:
    stories = [
        """Story: The Lost Key
Mira searched her flat for an hour. The key was not on the hook, not in her bag. She sat, breathed, and retraced her morning. She had dropped it beside the plant when bringing in post. Small panic, simple fix.""",
        """Story: The Night Train
Tom boarded a late train north. Rain tapped the windows. A stranger shared sandwiches and a map of hiking trails. By dawn, Tom had a plan for a weekend he had not expected.""",
        """Story: The Broken Bike
Leo's chain snapped two miles from home. An older cyclist stopped, taught him to link it, and rode slowly beside him until the path cleared. Leo learned repair and kindness the same afternoon.""",
        """Story: The Empty Stage
A singer forgot the first line. The crowd waited. She laughed, said the truth, and started again. The second try was warmer than the first would have been.""",
        """Story: The Garden
Every spring, Mrs. Chen planted tomatoes. Neighbors thought it too much work. By July, bowls of red fruit appeared on doorsteps. The garden fed the block, not just one table.""",
        """Story: The Letter
After years, Ana found a letter she never sent. Reading it, she saw how much she had grown. She wrote a new letter, shorter and kinder, and mailed it that week.""",
        """Story: The Storm
Wind knocked out power for two days. Families shared candles and stories. When lights returned, people kept visiting each other. The storm passed; the habit stayed.""",
        """Story: The Puppy
A small dog followed Sam home. The shelter said he had been lost a week. Sam adopted him. The dog learned trust slowly, one calm day at a time.""",
        """Story: The Exam
Jamal studied in short blocks with breaks. He failed a practice test, fixed weak topics, and slept well before the real one. Score was fine. Process mattered more than panic.""",
        """Story: The Bridge
Two towns argued about a bridge for decades. A flood washed the old one away. They built together in weeks. Shared work ended an old argument.""",
        """Story: The Lighthouse
Every storm, the keeper climbed wet steps to light the lamp. Ships steered clear. One night the bulb failed; dawn revealed she had rewired it in the dark.""",
        """Story: The Recipe Box
Grandma's handwriting faded on index cards. Maya typed each recipe and cooked one weekly. Flavor returned memories louder than photos.""",
        """Story: The Job Offer
Ravi had two offers: money or mission. He chose mission and smaller pay. Years later he led a team that fixed real problems.""",
        """Story: The Snow Day
Schools closed. Parents worked from kitchen tables. Kids built forts. Neighbors traded cocoa. Chaos became community for one day.""",
        """Story: The Old Guitar
Dust covered strings for a decade. One evening Eli tuned it badly and played anyway. Progress sounded better than perfection.""",
        """Story: The Market
Every Saturday Lina sold honey at the market. Regulars brought jars to refill. Trust grew sweeter than sugar.""",
        """Story: The Wrong Bus
Paolo boarded the wrong bus abroad. He met a guide who showed hidden gardens. Getting lost opened the best afternoon.""",
        """Story: The Volunteer Shift
At the shelter, Ken served soup and listened. A guest said thanks for seeing me. Ken returned every month.""",
        """Story: The Deadline
The report was due at midnight. They split tasks, checked facts twice, and finished early. Teamwork beat heroics.""",
    ]
    return "\n\n".join(stories)


def build_jokes() -> str:
    jokes = [
        "Joke: Why did the scarecrow win an award? He was outstanding in his field.",
        "Joke: Why don't scientists trust atoms? Because they make up everything.",
        "Joke: What do you call fake spaghetti? An impasta.",
        "Joke: Why did the bicycle fall over? It was two tired.",
        "Joke: What do you call a bear with no teeth? A gummy bear.",
        "Joke: Why did the math book look sad? It had too many problems.",
        "Joke: What did the ocean say to the beach? Nothing, it just waved.",
        "Joke: Why can't you give Elsa a balloon? She will let it go.",
        "Joke: What do you call cheese that is not yours? Nacho cheese.",
        "Joke: Why did the coffee file a police report? It got mugged.",
        "Joke: What do you call a sleeping bull? A bulldozer.",
        "Joke: Why did the student eat homework? The teacher said it was a piece of cake.",
        "Joke: What is brown and sticky? A stick.",
        "Joke: Why do bees have sticky hair? They use honeycombs.",
        "Joke: What did one wall say to the other? Meet you at the corner.",
        "Joke: Why did the golfer bring two pairs of pants? In case he got a hole in one.",
        "Joke: What do you call a can opener that does not work? A can't opener.",
        "Joke: Why are frogs happy? They eat whatever bugs them.",
        "Joke: What do you call a fish with no eyes? A fsh.",
        "Joke: Why did the picture go to jail? It was framed.",
        "Joke: What do you call a dinosaur that crashes his car? Tyrannosaurus wrecks.",
        "Joke: Why did the tomato turn red? Because it saw the salad dressing.",
        "Joke: What do you call a boomerang that won't come back? A stick.",
        "Joke: Why don't eggs tell jokes? They'd crack each other up.",
        "Joke: What do you call a factory that makes okay products? A satisfactory.",
        "Joke: Why did the cookie go to the hospital? Because it felt crummy.",
        "Joke: What do you call a pile of cats? A meowtain.",
        "Joke: Why did the stadium get hot after the game? All the fans left.",
        "Joke: What do you call a belt made of watches? A waist of time.",
        "Joke: Why did the banana go to the doctor? It wasn't peeling well.",
        "Joke: What do you call a snowman with a six-pack? An abdominal snowman.",
        "Joke: Why don't skeletons fight each other? They don't have the guts.",
        "Joke: What do you call a lazy kangaroo? A pouch potato.",
        "Joke: Why did the computer keep sneezing? It had a virus.",
    ]
    return "\n".join(jokes)


def build_life_qa() -> str:
    return _qa([
        ("I feel anxious.", "That is understandable. Try slow breathing and one small task you can finish."),
        ("I am lonely.", "Many people feel that way sometimes. Text one person or join a local group."),
        ("How do I focus?", "Remove one distraction, set a timer for 25 minutes, and take a short break."),
        ("I failed a test.", "Failure is feedback. Review mistakes and study in shorter sessions."),
        ("How do I make friends?", "Show up regularly where people share your interests and be kind."),
        ("Why is sleep important?", "Sleep repairs your body and helps memory and mood."),
        ("What is empathy?", "Empathy is understanding how someone else feels."),
        ("How do I handle stress?", "Exercise, sleep, and talking to someone you trust can help."),
        ("Tell me a short story.", "Once, a traveler lost a map and found a better path by asking locals."),
        ("Tell me a joke.", "Why did the computer go to the doctor? It had a virus."),
        ("What is psychology?", "Psychology is the science of mind and behavior."),
        ("What is history?", "History is the study of past human events and societies."),
        ("What is culture?", "Culture is shared beliefs, customs, and arts of a group."),
        ("What is philosophy?", "Philosophy asks big questions about knowledge, ethics, and existence."),
        ("What is economics?", "Economics studies how people produce, trade, and use resources."),
        ("What is geography?", "Geography studies places, landscapes, and how people interact with Earth."),
        ("What is literature?", "Literature is written works valued for art and ideas."),
        ("What is ethics?", "Ethics examines right and wrong in human action."),
        ("How do habits form?", "Repeated actions in stable contexts become automatic over time."),
        ("What helps memory?", "Sleep, repetition, and connecting new facts to things you know."),
        ("What is the scientific method?", "Ask a question, hypothesize, test, analyze, and conclude from evidence."),
        ("What is biology?", "Biology is the study of living organisms and life processes."),
        ("What is chemistry?", "Chemistry studies matter, atoms, molecules, and reactions."),
        ("What is physics?", "Physics studies matter, energy, force, and motion."),
        ("What is astronomy?", "Astronomy studies stars, planets, and the universe."),
        ("What is an ecosystem?", "An ecosystem is living things plus their environment interacting."),
        ("What is evolution?", "Evolution is change in inherited traits over generations."),
        ("What is a vaccine?", "A vaccine trains the immune system to recognize a pathogen safely."),
        ("What is democracy?", "Democracy is government where power comes from the people, often by voting."),
        ("Who was Albert Einstein?", "Einstein developed relativity and won a Nobel for explaining the photoelectric effect."),
        ("Who was Rosa Parks?", "Rosa Parks refused to give up her bus seat, sparking civil rights action."),
        ("What is 15 + 27?", "Fifteen plus twenty-seven equals forty-two."),
        ("What is 9 times 7?", "Nine times seven equals sixty-three."),
        ("What is half of 50?", "Half of fifty is twenty-five."),
        ("What is a metaphor?", "A metaphor describes something by saying it is something else for effect."),
        ("What is irony?", "Irony is when outcome differs from expectation, often with humor or insight."),
        ("How do I stay motivated?", "Set small visible goals and track progress weekly."),
        ("What is burnout?", "Burnout is exhaustion from prolonged stress, often work-related."),
        ("What is a boundary?", "A boundary is a limit you set to protect time, energy, or safety."),
        ("Recommend a morning habit.", "Drink water, stretch for two minutes, and plan one priority."),
        ("What is climate change?", "Climate change is long-term shift in global temperature and weather patterns."),
        ("What is renewable energy?", "Renewable energy comes from sources like sun, wind, and water that replenish."),
        ("What is the brain?", "The brain controls thought, memory, emotion, and movement via neurons."),
        ("What is culture shock?", "Culture shock is disorientation when entering a very different society."),
    ])


def build_multi_turn_life() -> str:
    convos = [
        [
            ("I had a rough day.", "Sorry to hear that. Want to talk about what happened?"),
            ("Work was overwhelming.", "That sounds draining. What part felt hardest?"),
            ("Too many emails.", "Batch them: check at set times instead of all day. Small boundaries help."),
        ],
        [
            ("Tell me something interesting.", "Honey never spoils. Archaeologists found edible honey in ancient tombs."),
            ("Really?", "Yes! Low water content and acidity preserve it for years."),
            ("Cool fact.", "Nature is full of surprises like that."),
        ],
        [
            ("I want to read more.", "Pick one short book or article to start."),
            ("I never finish.", "Ten pages a day beats zero. Consistency wins."),
            ("Good idea.", "You can build from there."),
        ],
        [
            ("What is mindfulness?", "Mindfulness is paying attention to the present without harsh judgment."),
            ("How do I try it?", "Sit quietly and notice your breath for two minutes."),
            ("Simple enough.", "Simple is the point."),
        ],
        [
            ("Who was Marie Curie?", "She pioneered research on radioactivity and won two Nobel Prizes."),
            ("Amazing.", "She worked with courage despite many barriers."),
            ("Inspiring.", "Her papers are still stored in lead boxes due to radiation."),
        ],
        [
            ("I feel stuck in life.", "Many people feel that at times. What area feels most stuck?"),
            ("My career.", "List skills you enjoy using. One conversation with someone in a field you like can open paths."),
            ("That helps.", "Small steps compound over months."),
        ],
        [
            ("Explain rain.", "Sun evaporates water; clouds form; droplets fall when heavy enough."),
            ("Why do we need rain?", "Rain fills rivers, lakes, and nourishes crops."),
            ("Makes sense.", "Water cycles constantly on Earth."),
        ],
        [
            ("Tell me a joke.", "Why did the math book look sad? Too many problems."),
            ("Ha.", "Humor makes learning lighter."),
            ("Got another?", "Why don't scientists trust atoms? They make up everything."),
        ],
        [
            ("What is philosophy?", "Philosophy examines knowledge, ethics, mind, and existence with reasoned argument."),
            ("Example?", "Asking what makes an action right is ethics, a branch of philosophy."),
            ("Interesting.", "Questions matter as much as answers."),
        ],
        [
            ("I'm learning Python.", "Great choice! Start with variables, loops, and functions."),
            ("It's hard.", "Practice twenty minutes daily. Errors teach you."),
            ("OK I will.", "Consistency beats cramming."),
        ],
        [
            ("What is black hole?", "A black hole has gravity so strong light cannot escape past a boundary."),
            ("Scary.", "They are far away and help scientists test physics."),
            ("Cool.", "Space holds many extremes."),
        ],
        [
            ("How do I apologize?", "Name what you did, acknowledge impact, and say how you'll change."),
            ("What if they won't forgive?", "You control sincerity; they control forgiveness."),
            ("Fair.", "Repair takes time."),
        ],
        [
            ("Best study tip?", "Quiz yourself instead of rereading passively."),
            ("Why?", "Retrieval strengthens memory pathways."),
            ("I'll try that.", "Active recall is powerful."),
        ],
        [
            ("What is art for?", "Art expresses feeling, beauty, and ideas beyond plain facts."),
            ("Can anyone make art?", "Yes. Skill grows with practice and play."),
            ("Nice.", "Creativity belongs to everyone."),
        ],
    ]
    return "\n\n".join(_format_conversation(c) for c in convos)


def build_science() -> str:
    return _articles([
        ("The Solar System", "Eight planets orbit the Sun. Rocky planets lie inner; gas giants outer. Moons, asteroids, and comets accompany them."),
        ("Stars", "Stars are giant balls of hot plasma. Our Sun is average-sized. Stars fuse hydrogen into helium, releasing energy."),
        ("Galaxies", "Galaxies contain billions of stars. The Milky Way is our spiral galaxy. Telescopes reveal countless others."),
        ("Plate Tectonics", "Earth's crust moves on slow plates. Collisions build mountains; separation forms oceans. Quakes and volcanoes mark boundaries."),
        ("The Water Cycle", "Water evaporates, forms clouds, falls as rain or snow, and flows back to seas. Life depends on this cycle."),
        ("Photosynthesis", "Plants convert light, water, and CO2 into sugar and oxygen. This feeds ecosystems and shapes our air."),
        ("Cells", "Cells are life's building blocks. They have membranes and machinery. Single cells can live alone; humans have trillions."),
        ("Genetics", "Genes are DNA segments coding traits. Inheritance passes genes from parents. Mutations introduce variation."),
        ("The Immune System", "White blood cells attack pathogens. Memory cells remember past invaders. Vaccines train this memory safely."),
        ("Bacteria and Viruses", "Bacteria are single-cell organisms; some help us, some harm. Viruses need hosts to reproduce. Handwashing reduces spread."),
        ("The Periodic Table", "Elements list by atomic number. Columns share traits. Hydrogen and oxygen form water when bonded."),
        ("Chemical Bonds", "Atoms bond by sharing or transferring electrons. Strong bonds hold molecules together. Reactions rearrange bonds."),
        ("Energy Types", "Energy appears as motion, light, heat, and stored forms. It converts but total energy is conserved."),
        ("Sound", "Sound is vibration traveling through matter. Pitch relates to frequency. Volume relates to amplitude."),
        ("Light", "Light travels fast in vacuum. Prisms split white light into colors. Eyes detect a narrow band of wavelengths."),
        ("Electricity Basics", "Charge flows through conductors. Voltage pushes; resistance limits. Circuits need closed paths."),
        ("Magnets", "Magnets have north and south poles. Moving charges create magnetic fields. Compasses align with Earth's field."),
        ("Machines", "Levers, wheels, and pulleys trade force for distance. Engines convert fuel into motion. Efficiency is never perfect."),
        ("Measurement", "Science uses standard units: meters, kilograms, seconds. Tools must be calibrated. Uncertainty is reported honestly."),
        ("Scientific Ethics", "Scientists should avoid fraud and harm. Peer review checks claims. Experiments on people need consent."),
    ])


def build_math_practical() -> str:
    examples = [
        "Counting: 1, 2, 3, 4, 5 helps children learn order.",
        "Addition: 8 + 7 = 15",
        "Subtraction: 20 - 13 = 7",
        "Multiplication: 11 * 11 = 121",
        "Division: 81 / 9 = 9",
        "Fractions: 1/3 + 1/3 = 2/3",
        "Decimals: 0.25 equals one quarter",
        "Percent: 20% of 150 = 30",
        "Ratio: 2:3 means two parts to three parts",
        "Average: mean of 4, 6, 10 is 20/3 about 6.67",
        "Area rectangle: 8 * 5 = 40 square units",
        "Area triangle: 1/2 * 6 * 4 = 12",
        "Circumference: 2 * pi * 3 about 18.85",
        "Order of ops: 3 + 4 * 2 = 11",
        "Algebra: if 2x = 18 then x = 9",
        "Word problem: 3 pencils at 2 dollars each cost 6 dollars",
        "Tip: 15% of 40 = 6 dollars tip",
        "Speed: 120 miles in 2 hours = 60 mph",
        "Conversion: 1 mile about 1.6 kilometers",
        "Probability: fair die P(6) = 1/6",
        "Budget: income 2000 minus rent 800 leaves 1200",
        "Interest simple: 1000 at 5% one year = 50 interest",
        "Geometry: square has four equal sides",
        "Angles: straight line is 180 degrees",
        "Negative numbers: -3 + 5 = 2",
    ]
    lines = ["Practical math:"]
    lines.extend(f"Example: {e}" for e in examples)
    return "\n".join(lines)


def build_philosophy() -> str:
    return _articles([
        ("Ethics", "Ethics asks what we ought to do. Consequences, duties, and virtues offer different lenses."),
        ("Logic", "Logic studies valid reasoning. Premises support conclusions. Fallacies look persuasive but fail logically."),
        ("Free Will", "Philosophers debate whether choices are truly free or determined by prior causes."),
        ("Knowledge", "Epistemology asks how we know things. Evidence, testimony, and reason all play roles."),
        ("Meaning of Life", "People find meaning in relationships, work, faith, or curiosity. There is no single answer for all."),
        ("Justice", "Justice concerns fair treatment and fair rules. Laws attempt justice but societies disagree on details."),
        ("Happiness", "Happiness mixes pleasure, purpose, and peace. Lasting happiness often ties to connection and growth."),
        ("Truth", "Truth aligns with facts or reality. Honesty builds trust. Misinformation spreads when people share without checking."),
        ("Stoicism", "Stoics focus on what you control: judgments and actions. External events are not fully ours to command."),
        ("Existentialism", "Existentialists stress freedom and responsibility. We shape life through choices even amid uncertainty."),
    ])


def build_technology() -> str:
    return _articles([
        ("Computers", "Computers process data with chips and memory. Software gives instructions hardware executes. They transform work, play, and communication."),
        ("The Web", "The World Wide Web links documents with URLs. Browsers fetch pages; servers host them. HTTPS encrypts traffic for safer browsing."),
        ("Smartphones", "Smartphones combine phone, camera, and computer. Apps extend capability. Battery and privacy are common user concerns."),
        ("Artificial Intelligence", "AI systems learn patterns from data. Uses include search, translation, and assistants. Good design needs fairness and safety."),
        ("Robotics", "Robots sense, decide, and act in the physical world. Factories, medicine, and homes use them. Humans still design goals and limits."),
        ("Programming", "Programming writes instructions computers follow. Languages like Python emphasize readability. Debugging fixes errors step by step."),
        ("Databases", "Databases store structured records. Queries retrieve filtered data. Backups protect against loss."),
        ("Cybersecurity", "Cybersecurity protects systems from theft and damage. Strong passwords and updates reduce risk. Phishing tricks users into revealing secrets."),
        ("Cloud Computing", "Cloud services run on remote servers you access online. Scaling and backups become easier. Trust the provider's security practices."),
        ("3D Printing", "3D printers build objects layer by layer from digital models. Prototyping and custom parts benefit. Materials range from plastic to metal."),
    ])


def build_history_people() -> str:
    return _articles([
        ("Leonardo da Vinci", "Leonardo painted the Mona Lisa and sketched flying machines. He blended art and science in Renaissance Italy."),
        ("Galileo Galilei", "Galileo used telescopes to support heliocentrism. His trial showed tension between science and authority."),
        ("Isaac Newton", "Newton described gravity and laws of motion. His work launched classical physics."),
        ("Charles Darwin", "Darwin proposed natural selection explaining diversity of life. Evidence came from fossils and observation."),
        ("Marie Curie", "Curie studied radioactivity and won two Nobel Prizes. She advanced medicine and chemistry."),
        ("Mahatma Gandhi", "Gandhi led nonviolent resistance to British rule in India. His methods inspired civil rights movements."),
        ("Martin Luther King Jr.", "King advocated civil rights through peaceful protest in the United States. His dream speech is famous worldwide."),
        ("Nelson Mandela", "Mandela fought apartheid in South Africa and later became president. He modeled reconciliation."),
        ("Cleopatra", "Cleopatra ruled Egypt and allied with Roman leaders. Her story mixes politics and legend."),
        ("Genghis Khan", "Genghis Khan united Mongol tribes and built a vast empire. Trade routes spread under Mongol peace."),
        ("Joan of Arc", "Joan led French forces during the Hundred Years War. She became a symbol of courage and faith."),
        ("Thomas Edison", "Edison patented the practical light bulb and phonograph. Inventors build on many failed trials."),
        ("Ada Lovelace", "Lovelace wrote early notes on computing machines. She is honored as a programming pioneer."),
        ("Alan Turing", "Turing formalized computation and helped break wartime codes. His work underpins computer science."),
        ("Jane Goodall", "Goodall studied wild chimpanzees for decades. Her research changed how we see animals."),
    ])


def build_geography() -> str:
    return _articles([
        ("Continents", "Earth has seven continents: Africa, Antarctica, Asia, Europe, North America, Oceania, South America. Cultures and climates vary widely."),
        ("Rivers", "Rivers carve land and supply water. Nile, Amazon, and Mississippi shaped civilizations. Dams store water and generate power."),
        ("Mount Everest", "Everest is Earth's highest peak in the Himalayas. Climbers face cold and thin air. Sherpa guides know the routes."),
        ("Sahara Desert", "The Sahara spans North Africa with extreme heat and dryness. Oases support life. Deserts can expand or shrink over time."),
        ("Mediterranean", "The Mediterranean Sea borders Europe, Africa, and Asia. Trade and empires flourished on its shores."),
        ("Great Barrier Reef", "Off Australia, this reef system hosts coral and fish. Warming oceans threaten coral bleaching."),
        ("Arctic", "The Arctic surrounds the North Pole with ice and tundra. Indigenous peoples adapted to cold for thousands of years."),
        ("Pacific Ocean", "The Pacific is the largest ocean. Ring of Fire volcanoes rim its edges. Islands dot its vast blue."),
        ("Europe", "Europe has many nations and languages. The EU promotes economic cooperation among members."),
        ("Brazil", "Brazil is the largest South American country. Amazon rainforest and vibrant cities define it."),
    ])


def build_health_wellness() -> str:
    return _articles([
        ("Nutrition", "Balanced meals include vegetables, protein, whole grains, and healthy fats. Water beats sugary drinks for hydration."),
        ("Exercise", "Aim for regular movement most days. Strength and cardio both help heart and bones."),
        ("Mental Health", "Mental health matters as much as physical health. Talk, therapy, and medicine can all help when struggling."),
        ("Hydration", "Water supports every organ. Thirst signals need. More needed in heat and exercise."),
        ("Posture", "Neutral spine reduces back strain. Screens tempt slouching; breaks and setup help."),
        ("First Aid", "Clean minor cuts, apply pressure to stop bleeding, and know when to call emergency services."),
        ("Sleep Hygiene", "Dark, cool rooms and consistent bedtimes improve sleep. Limit caffeine late in the day."),
        ("Sun Safety", "Sunscreen and shade reduce skin damage. UV is strong even on cloudy days."),
        ("Dental Care", "Brush twice daily and floss. Regular checkups prevent painful problems."),
        ("Stretching", "Gentle stretches maintain flexibility. Hold stretches without bouncing."),
    ])


def build_daily_tips() -> str:
    tips = [
        "Tip: Write tomorrow's top three tasks before bed.",
        "Tip: Label files clearly so future you finds them fast.",
        "Tip: Walk ten minutes after meals to aid digestion and mood.",
        "Tip: Keep a reusable water bottle visible on your desk.",
        "Tip: Unsubscribe from emails you never read.",
        "Tip: Learn one new word daily and use it in conversation.",
        "Tip: Thank someone sincerely each day.",
        "Tip: Put your phone in another room while focusing.",
        "Tip: Batch similar errands into one trip.",
        "Tip: Review subscriptions monthly; cancel unused ones.",
        "Tip: Cook extra dinner for easy lunch tomorrow.",
        "Tip: Stand and stretch every hour when desk working.",
        "Tip: Keep emergency cash small but accessible.",
        "Tip: Photograph important documents and store securely.",
        "Tip: Introduce yourself to one neighbor you do not know.",
        "Tip: Read ingredient lists; shorter often means simpler.",
        "Tip: Set a consistent wake time even on weekends.",
        "Tip: Use a password manager instead of repeating passwords.",
        "Tip: Plant herbs on a windowsill for fresh flavor.",
        "Tip: Listen fully before planning your reply.",
    ]
    return "\n".join(tips)


def build_more_stories() -> str:
    stories = [
        """Story: The Clock Repair
Hassan fixed clocks for forty years. A child brought a broken watch from her grandmother. Inside was a tiny note: Be patient. Hassan smiled and taught her to wind it gently.""",
        """Story: The Ferry
Each dawn the ferry crossed the river. One foggy morning the captain slowed for drifting logs. Passengers arrived late but safe. Speed was not the only virtue.""",
        """Story: The Mentor
A senior engineer stayed late to review Nina's code. No lecture, just questions. Nina found the bug herself. Respect grew both ways.""",
        """Story: The Bake Sale
The school needed funds. Parents argued recipes. They combined ideas: one table, many flavors. The line stretched down the block.""",
        """Story: The Hike
On the trail, Mia sprained an ankle. Strangers shared bandages and walked her down. She learned wilderness is also kindness.""",
        """Story: The Translation
A refugee needed forms in a new language. A librarian helped for free each Tuesday. Months later he returned fluent, helping others.""",
        """Story: The Orchestra
First rehearsal was messy. The conductor split parts slowly. Concert night sounded unified. Practice turned noise into music.""",
        """Story: The Inheritance
Two siblings fought over a house. They sold it, split funds, and traveled together. Memory mattered more than walls.""",
        """Story: The Raincoat
Jon forgot rain gear. A coworker lent a spare. Jon bought two extras kept at the office. Small cycles of care spread.""",
        """Story: The Telescope
Grandpa showed constellations from the porch. Years later she built satellites. Wonder started with shared sky.""",
        """Story: The apology
They had not spoken in years. One sent a short note owning their part. Coffee followed. Friendship rebuilt slowly.""",
        """Story: The marathon
Training hurt knees and pride. She walked parts, finished anyway. The medal meant showing up.""",
        """Story: The library card
At twelve, books were free transport. At thirty, she published one. Access opened worlds.""",
        """Story: The power cut
Neighbors grilled food before it spoiled. Music played from a battery radio. Darkness became a block party.""",
        """Story: The seed
A cracked sidewalk held a dandelion. A kid watered it daily. City workers mowed around it one summer.""",
    ]
    return "\n\n".join(stories)


def build_more_jokes() -> str:
    jokes = [
        "Joke: What do you call a bear with no ears? B.",
        "Joke: Why can't a leopard hide? Because he is always spotted.",
        "Joke: What do you call a dog magician? A labracadabrador.",
        "Joke: Why did the math teacher love geometry? It was just her angle.",
        "Joke: What do you call an alligator in a vest? An investigator.",
        "Joke: Why did the kid cross the playground? To get to the other slide.",
        "Joke: What do you call a sleeping dinosaur? A dino-snore.",
        "Joke: Why did the cookie visit the doctor? It felt crummy.",
        "Joke: What do you call a cow with no legs? Ground beef.",
        "Joke: Why did the student study on an airplane? He wanted higher education.",
        "Joke: What do you call a fish that sings? Auto-tuna.",
        "Joke: Why was the broom late? It swept in.",
        "Joke: What do you call a bee that can't make up its mind? A maybe.",
        "Joke: Why did the picture go to jail? It was framed.",
        "Joke: What do you call a duck that steals? A robber ducky.",
        "Joke: Why don't oysters share? Because they are shellfish.",
        "Joke: What do you call a cold dog? A chili dog.",
        "Joke: Why did the scarecrow win? Outstanding in his field.",
        "Joke: What do you call a boomerang that works? A stick.",
        "Joke: Why did the computer go to art school? It had graphic design.",
    ]
    return "\n".join(jokes)


def build_more_qa() -> str:
    return _qa([
        ("Who was Napoleon?", "Napoleon Bonaparte was a French military leader and emperor who reshaped Europe."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("What is the capital of Germany?", "The capital of Germany is Berlin."),
        ("What is the capital of Canada?", "The capital of Canada is Ottawa."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
        ("What is H2O?", "H2O is the chemical formula for water."),
        ("What is CO2?", "CO2 is carbon dioxide, a gas plants use and humans exhale."),
        ("What is oxygen?", "Oxygen is a gas we breathe; it supports combustion and life."),
        ("What is a mammal?", "Mammals are warm-blooded vertebrates that usually feed milk to young."),
        ("What is a reptile?", "Reptiles include snakes and lizards; they are cold-blooded and scaly."),
        ("What is a bird?", "Birds have feathers, lay eggs, and most can fly."),
        ("What is a fish?", "Fish live in water and breathe through gills."),
        ("What is a continent?", "A continent is a large continuous landmass on Earth."),
        ("What is a volcano?", "A volcano releases molten rock and gas from Earth's interior."),
        ("What is an earthquake?", "An earthquake shakes ground when rocks suddenly move along faults."),
        ("What is a hurricane?", "A hurricane is a large rotating tropical storm with strong winds."),
        ("What is snow?", "Snow is frozen precipitation made of ice crystals."),
        ("What is thunder?", "Thunder is the sound of air expanding rapidly after lightning heats it."),
        ("What is a rainbow?", "A rainbow forms when light refracts and reflects in water droplets."),
        ("What is iron?", "Iron is a strong metal used in steel; symbol Fe on periodic table."),
        ("What is gold?", "Gold is a soft yellow metal valued for jewelry and electronics."),
        ("What is salt?", "Table salt is sodium chloride; it flavors food and melts ice on roads."),
        ("What is sugar?", "Sugar is a sweet carbohydrate; glucose fuels cells."),
        ("What is protein?", "Proteins are molecules built from amino acids; muscles and enzymes need them."),
        ("What is a calorie?", "A calorie measures energy in food."),
        ("What is a year?", "A year is the time Earth takes to orbit the Sun, about 365 days."),
        ("What is a leap year?", "A leap year adds one day in February every four years to keep calendars aligned."),
        ("What is a decade?", "A decade is ten years."),
        ("What is a century?", "A century is one hundred years."),
        ("What is midnight?", "Midnight is twelve o clock at night, start of a new day."),
        ("How many days in a week?", "There are seven days in a week."),
        ("How many months in a year?", "There are twelve months in a year."),
        ("What is a prime number?", "A prime number has exactly two factors: one and itself."),
        ("What is a square number?", "A square number is n times n, like 9 is 3 squared."),
        ("What is a cube?", "A cube has six equal square faces."),
        ("What is a sphere?", "A sphere is a round 3D shape like a ball."),
        ("What is pi?", "Pi is the ratio of a circle circumference to its diameter, about 3.14."),
        ("What is a triangle?", "A triangle has three sides and three angles summing to 180 degrees."),
        ("What is a rectangle?", "A rectangle has four sides with opposite sides equal and four right angles."),
        ("What is kindness?", "Kindness is choosing actions that help or comfort others."),
        ("What is courage?", "Courage is doing what is right despite fear."),
        ("What is honesty?", "Honesty means telling the truth and acting sincerely."),
        ("What is patience?", "Patience is waiting calmly without unnecessary anger."),
        ("What is gratitude?", "Gratitude is thankful recognition of good things."),
        ("What is respect?", "Respect treats people and rules with appropriate regard."),
    ])


def build_more_multi_turn() -> str:
    convos = [
        [
            ("Where is France?", "France is in western Europe. Its capital is Paris."),
            ("What is it known for?", "Art, cuisine, history, and landmarks like the Eiffel Tower."),
            ("Nice.", "France shaped much of modern European culture."),
        ],
        [
            ("I'm nervous about an interview.", "Prepare three examples of work you are proud of."),
            ("What if I blank?", "Pause, breathe, and ask to revisit the question."),
            ("Good tip.", "Interviewers expect some nerves."),
        ],
        [
            ("Explain tides.", "Moon gravity pulls ocean water, creating high and low tides."),
            ("And the Sun?", "The Sun also affects tides, but less than the Moon."),
            ("Cool.", "Coastal life follows tidal rhythms."),
        ],
        [
            ("Recommend a book habit.", "Keep a book where you usually scroll your phone."),
            ("Simple.", "Environment beats willpower."),
            ("I'll try.", "Even five pages count."),
        ],
        [
            ("What is jazz?", "Jazz is music with improvisation, swing, and rich harmony, born in America."),
            ("Famous artist?", "Louis Armstrong helped popularize jazz worldwide."),
            ("Thanks.", "Music history is human history."),
        ],
        [
            ("My friend is sad.", "Listen more than you fix. Ask if they want advice or company."),
            ("That's hard.", "Presence alone often helps."),
            ("OK.", "Check in again later."),
        ],
        [
            ("What is coding?", "Coding writes instructions computers execute precisely."),
            ("Is it hard?", "It is learnable like any language with practice."),
            ("Encouraging.", "Start tiny and build."),
        ],
        [
            ("Why exercise?", "It strengthens heart, bones, mood, and sleep quality."),
            ("How much?", "Even thirty minutes most days helps."),
            ("Fair.", "Movement is medicine for many."),
        ],
        [
            ("What is Shakespeare?", "An English playwright from the 1500s and 1600s."),
            ("Famous play?", "Hamlet asks deep questions about action and doubt."),
            ("Classic.", "His phrases still appear in English."),
        ],
        [
            ("I procrastinate.", "Name the smallest next step and do only that."),
            ("Like open the doc?", "Exactly. Starting is the hardest part."),
            ("Will do.", "Momentum follows action."),
        ],
        [
            ("What is Bitcoin?", "Bitcoin is digital currency recorded on a decentralized ledger called blockchain."),
            ("Is it risky?", "Prices swing; only risk what you can afford to lose."),
            ("Got it.", "Learn before investing."),
        ],
        [
            ("Tell me about dogs.", "Dogs were domesticated from wolves thousands of years ago."),
            ("Why loyal?", "They evolved alongside humans as partners."),
            ("Sweet.", "Training uses clear consistent cues."),
        ],
        [
            ("What is meditation?", "Meditation trains attention and calm through focused practice."),
            ("How long?", "Even five minutes daily can help beginners."),
            ("I'll start small.", "That is wise."),
        ],
        [
            ("Explain seasons.", "Earth's tilt causes seasons as we orbit the Sun."),
            ("Why winter cold?", "Sunlight hits your hemisphere at a lower angle in winter."),
            ("Makes sense.", "Tilt explains a lot."),
        ],
        [
            ("I want to write.", "Write badly first; editing comes later."),
            ("Daily?", "Daily pages build skill faster than waiting for inspiration."),
            ("Deal.", "Writers write."),
        ],
    ]
    return "\n\n".join(_format_conversation(c) for c in convos)


def build_knowledge_corpus() -> str:
    wiki = _expand(build_general_wiki(), 4, "wiki")
    psych = _expand(build_psychology(), 4, "psychology")
    life = _expand(build_human_experience(), 4, "life")
    science = _expand(build_science(), 4, "science")
    math = _expand(build_math_practical(), 5, "math")
    philosophy = _expand(build_philosophy(), 3, "philosophy")
    tech = _expand(build_technology(), 3, "technology")
    people = _expand(build_history_people(), 3, "people")
    geo = _expand(build_geography(), 3, "geography")
    health = _expand(build_health_wellness(), 3, "health")
    tips = _expand(build_daily_tips(), 5, "tips")
    stories = _expand(build_stories() + "\n\n" + build_more_stories(), 3, "stories")
    jokes = _expand(build_jokes() + "\n" + build_more_jokes(), 4, "jokes")
    qa = _expand(build_life_qa() + build_more_qa(), 3, "life-qa")
    multi = _expand(build_multi_turn_life() + "\n\n" + build_more_multi_turn(), 6, "life-chat")
    return (
        section("General Knowledge", wiki)
        + section("Psychology", psych)
        + section("Human Experience", life)
        + section("Science", science)
        + section("Practical Math", math)
        + section("Philosophy", philosophy)
        + section("Technology", tech)
        + section("Historical Figures", people)
        + section("Geography", geo)
        + section("Health and Wellness", health)
        + section("Daily Tips", tips)
        + section("Short Stories", stories)
        + section("Jokes", jokes)
        + section("Life Q&A", qa)
        + section("Multi-turn Life Chat", multi)
    )
