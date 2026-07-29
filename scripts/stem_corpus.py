"""Science, math, physics, and multi-turn chat training data."""

from scripts.build_training_data import section


def _format_conversation(turns: list[tuple[str, str]]) -> str:
    lines = []
    for user, assistant in turns:
        lines.append(f"User: {user}\nAssistant: {assistant}")
    return "\n".join(lines) + "\n"


def build_physics_articles() -> str:
    articles = [
        (
            "Newton's First Law",
            "Newton's first law says an object at rest stays at rest, and an object in motion "
            "stays in motion at constant speed, unless a force acts on it. This is also called inertia.",
        ),
        (
            "Newton's Second Law",
            "Newton's second law links force, mass, and acceleration: F equals m times a. "
            "A bigger force causes greater acceleration. A bigger mass resists acceleration more.",
        ),
        (
            "Newton's Third Law",
            "For every action there is an equal and opposite reaction. When you push on a wall, "
            "the wall pushes back on you with the same strength in the opposite direction.",
        ),
        (
            "Speed and Velocity",
            "Speed is how fast something moves. Velocity is speed with direction. "
            "A car going 60 mph north has a different velocity than one going 60 mph south.",
        ),
        (
            "Acceleration",
            "Acceleration is the rate of change of velocity. Braking, turning, and speeding up "
            "all count as acceleration. It is measured in meters per second squared.",
        ),
        (
            "Friction",
            "Friction is a force that opposes motion when surfaces rub together. "
            "It helps shoes grip the ground but also slows sliding objects.",
        ),
        (
            "Energy",
            "Energy is the capacity to do work. Kinetic energy comes from motion. "
            "Potential energy is stored, like a raised weight or stretched spring.",
        ),
        (
            "Conservation of Energy",
            "Energy cannot be created or destroyed, only converted from one form to another. "
            "A rolling ball may lose speed to heat from friction, but total energy is conserved.",
        ),
        (
            "Waves",
            "A wave carries energy through a medium or space. Sound is a wave in air. "
            "Light is an electromagnetic wave. Waves have wavelength, frequency, and amplitude.",
        ),
        (
            "Electric Charge",
            "Matter contains positive and negative charges. Like charges repel; opposites attract. "
            "Static electricity builds when charges separate, such as after rubbing a balloon.",
        ),
        (
            "Current and Voltage",
            "Electric current is the flow of charge through a wire. Voltage is the push that drives "
            "current. Resistance limits current, as described by Ohm's law: V equals I times R.",
        ),
        (
            "Magnetism",
            "Magnets have north and south poles. Like poles repel; unlike poles attract. "
            "Moving charges create magnetic fields. Earth has a magnetic field from its core.",
        ),
        (
            "Light and Color",
            "White light contains many colors. A prism separates them by wavelength. "
            "Objects appear colored because they reflect some wavelengths and absorb others.",
        ),
        (
            "Heat and Temperature",
            "Temperature measures average particle motion. Heat is energy transferred due to "
            "a temperature difference. Heat flows from hot to cold until balance is reached.",
        ),
        (
            "Pressure",
            "Pressure is force divided by area. Atmospheric pressure is the weight of air above us. "
            "Pressure increases with depth in a fluid because more fluid sits above.",
        ),
        (
            "Buoyancy",
            "An object in fluid feels an upward buoyant force. If buoyancy exceeds weight, "
            "the object floats. Ships float because they displace enough water to balance their weight.",
        ),
        (
            "Relativity Basics",
            "Einstein's special relativity shows that time and space depend on motion. "
            "Nothing with mass can reach the speed of light. E equals m c squared links mass and energy.",
        ),
        (
            "Quantum Basics",
            "At tiny scales, particles behave like waves and packets called quanta. "
            "Electrons in atoms occupy discrete energy levels rather than any orbit.",
        ),
        (
            "The Atom",
            "Atoms have a dense nucleus of protons and neutrons, surrounded by electrons. "
            "The number of protons defines the element. Atoms combine to form molecules.",
        ),
        (
            "Radioactivity",
            "Some nuclei are unstable and decay, emitting radiation. Alpha, beta, and gamma "
            "radiation differ in penetrating power. Half-life measures decay rate.",
        ),
    ]
    return "\n\n".join(f"Article: {title}\n{body}" for title, body in articles)


def build_math_articles() -> str:
    articles = [
        (
            "Addition",
            "Addition combines quantities into a total. The symbol is plus. "
            "For example, 3 plus 5 equals 8. Addition is commutative: 3 plus 5 equals 5 plus 3.",
        ),
        (
            "Subtraction",
            "Subtraction finds the difference between numbers. It is the inverse of addition. "
            "For example, 10 minus 4 equals 6.",
        ),
        (
            "Multiplication",
            "Multiplication is repeated addition. 4 times 3 means 4 added three times: 4 plus 4 plus 4 equals 12. "
            "The result is called the product.",
        ),
        (
            "Division",
            "Division splits a number into equal parts. 12 divided by 3 equals 4 because three groups of four make twelve. "
            "Division by zero is undefined.",
        ),
        (
            "Fractions",
            "A fraction shows part of a whole. The top is the numerator; the bottom is the denominator. "
            "One half means one part of two equal parts.",
        ),
        (
            "Decimals",
            "Decimals use a point to show parts less than one. 0.5 equals one half. "
            "Each place to the right is one tenth of the previous place.",
        ),
        (
            "Percentages",
            "A percent means parts per hundred. 25 percent equals 25 out of 100, or one quarter. "
            "To find 20 percent of 50, multiply 0.20 times 50 equals 10.",
        ),
        (
            "Order of Operations",
            "Evaluate parentheses first, then exponents, then multiplication and division left to right, "
            "then addition and subtraction left to right. Example: 2 plus 3 times 4 equals 14, not 20.",
        ),
        (
            "Algebra",
            "Algebra uses letters for unknown values. Solving an equation means finding the value "
            "that makes both sides equal. If x plus 3 equals 7, then x equals 4.",
        ),
        (
            "Linear Equations",
            "A linear equation graphs as a straight line. Form y equals m x plus b, where m is slope "
            "and b is the y-intercept. Slope measures steepness.",
        ),
        (
            "Area",
            "Area measures surface size. Rectangle area equals length times width. "
            "Triangle area equals one half times base times height.",
        ),
        (
            "Volume",
            "Volume measures three-dimensional space. Box volume equals length times width times height. "
            "Volume is measured in cubic units.",
        ),
        (
            "Pythagorean Theorem",
            "In a right triangle, a squared plus b squared equals c squared, where c is the hypotenuse. "
            "If legs are 3 and 4, the hypotenuse is 5.",
        ),
        (
            "Prime Numbers",
            "A prime has exactly two factors: 1 and itself. Examples: 2, 3, 5, 7, 11. "
            "The number 1 is not prime. Every integer above 1 factors uniquely into primes.",
        ),
        (
            "Ratios and Proportions",
            "A ratio compares two quantities. A proportion states two ratios are equal. "
            "If 2 over 3 equals x over 6, then x equals 4.",
        ),
        (
            "Statistics Basics",
            "The mean is the average. The median is the middle value when sorted. "
            "The mode is the most frequent value. Range is max minus min.",
        ),
        (
            "Probability",
            "Probability measures likelihood from 0 to 1. Fair coin flip: heads probability is one half. "
            "Add probabilities of mutually exclusive outcomes to get total probability.",
        ),
        (
            "Angles",
            "A full turn is 360 degrees. A right angle is 90 degrees. "
            "Complementary angles sum to 90. Supplementary angles sum to 180.",
        ),
        (
            "Circles",
            "Circumference is 2 pi r. Area is pi r squared. Pi is about 3.14159. "
            "Diameter is twice the radius.",
        ),
        (
            "Scientific Notation",
            "Scientific notation writes large numbers compactly. 3000000 equals 3 times 10 to the 6th power. "
            "It makes very big and very small numbers easier to compare.",
        ),
    ]
    return "\n\n".join(f"Article: {title}\n{body}" for title, body in articles)


def build_chemistry_biology() -> str:
    articles = [
        (
            "The Cell",
            "The cell is the basic unit of life. All living things are made of one or more cells. "
            "Cells have a membrane, cytoplasm, and genetic material.",
        ),
        (
            "Photosynthesis Review",
            "Plants use sunlight, water, and carbon dioxide to make glucose and oxygen. "
            "The equation balances energy capture with chemical storage.",
        ),
        (
            "The Periodic Table Review",
            "Elements are organized by atomic number. Rows are periods; columns are groups with similar traits. "
            "Metals, nonmetals, and metalloids have different properties.",
        ),
        (
            "Chemical Reactions",
            "In a reaction, bonds break and form to make new substances. "
            "Reactants transform into products. Mass is conserved in a closed system.",
        ),
        (
            "Acids and Bases",
            "Acids donate hydrogen ions in water. Bases accept them. pH scale runs 0 to 14. "
            "Seven is neutral. Lower pH is more acidic; higher is more basic.",
        ),
        (
            "Human Heart",
            "The heart pumps blood through the body. It has four chambers. "
            "Arteries carry blood away from the heart; veins return blood toward it.",
        ),
        (
            "Immune System",
            "The immune system defends against pathogens. White blood cells attack invaders. "
            "Vaccines train the immune system to recognize threats.",
        ),
        (
            "Ecosystems",
            "An ecosystem includes living organisms and their environment. Energy flows from producers "
            "to consumers. Decomposers recycle nutrients.",
        ),
        (
            "States of Matter",
            "Matter commonly exists as solid, liquid, or gas. Heating adds energy and can cause phase changes. "
            "Plasma is ionized gas, common in stars.",
        ),
        (
            "Molecules and Compounds",
            "A molecule is two or more atoms bonded together. Water is H2O: two hydrogen, one oxygen. "
            "A compound has more than one element chemically combined.",
        ),
    ]
    return "\n\n".join(f"Article: {title}\n{body}" for title, body in articles)


def build_math_examples() -> str:
    examples = [
        "2 + 2 = 4",
        "3 + 5 = 8",
        "10 - 4 = 6",
        "12 - 7 = 5",
        "4 * 3 = 12",
        "6 * 7 = 42",
        "8 * 9 = 72",
        "15 / 3 = 5",
        "20 / 4 = 5",
        "100 / 25 = 4",
        "1/2 + 1/2 = 1",
        "1/4 + 1/4 = 1/2",
        "3/4 - 1/4 = 1/2",
        "2 * (3 + 4) = 14",
        "10 + 5 * 2 = 20",
        "(10 + 5) * 2 = 30",
        "x + 5 = 12, so x = 7",
        "2x = 10, so x = 5",
        "3x + 2 = 11, so x = 3",
        "y = 2x + 1 passes through (0, 1)",
        "Area of 5 by 4 rectangle = 20 square units",
        "Triangle base 6 height 3: area = 9",
        "Circle radius 2: area about 12.57",
        "25% of 80 = 20",
        "10% of 150 = 15",
        "Mean of 2, 4, 6 is 4",
        "Probability of rolling 6 on fair die = 1/6",
        "Speed = distance / time. 100 km in 2 h gives 50 km/h",
        "F = m * a. Force 10 N on 2 kg gives a = 5 m/s^2",
        "Kinetic energy = 1/2 m v^2",
        "Water boils at 100 degrees C at sea level",
        "Freezing point of water = 0 degrees C",
        "One kilometer = 1000 meters",
        "One hour = 60 minutes = 3600 seconds",
    ]
    lines = ["Math and physics examples:"]
    for ex in examples:
        lines.append(f"Example: {ex}")
    return "\n".join(lines)


def build_stem_qa() -> str:
    pairs = [
        ("What is force?", "Force is a push or pull that can change motion."),
        ("What is mass?", "Mass measures how much matter an object contains."),
        ("What is weight?", "Weight is the force of gravity on an object."),
        ("What is velocity?", "Velocity is speed with direction."),
        ("What is acceleration?", "Acceleration is the rate of change of velocity."),
        ("State Newton's second law.", "Force equals mass times acceleration."),
        ("What is friction?", "Friction opposes motion between surfaces in contact."),
        ("What is kinetic energy?", "Kinetic energy is energy of motion."),
        ("What is potential energy?", "Potential energy is stored energy, like height or a spring."),
        ("What is Ohm's law?", "Voltage equals current times resistance."),
        ("What is a photon?", "A photon is a particle of light."),
        ("What is an atom?", "An atom is the smallest unit of an element retaining its properties."),
        ("What is a molecule?", "A molecule is two or more atoms bonded together."),
        ("What is pH?", "pH measures how acidic or basic a solution is."),
        ("What is DNA for?", "DNA stores genetic instructions for life."),
        ("What is 7 times 8?", "Seven times eight equals fifty-six."),
        ("What is 144 divided by 12?", "One hundred forty-four divided by twelve equals twelve."),
        ("What is the square root of 81?", "The square root of eighty-one is nine."),
        ("What is pi approximately?", "Pi is approximately 3.14159."),
        ("What is 15 percent of 200?", "Fifteen percent of two hundred is thirty."),
        ("Solve x + 4 = 9.", "Subtract four from both sides: x equals five."),
        ("What is the area of a 3 by 7 rectangle?", "Three times seven equals twenty-one square units."),
        ("What is the Pythagorean theorem?", "In a right triangle, a squared plus b squared equals c squared."),
        ("What is a prime number?", "A prime number has exactly two factors: one and itself."),
        ("What is the mean of 3, 5, and 7?", "The mean is five."),
        ("What is absolute zero?", "Absolute zero is about minus 273 degrees Celsius, the coldest possible temperature."),
        ("What is the speed of light?", "Light travels about 300000 kilometers per second in vacuum."),
        ("What is density?", "Density equals mass divided by volume."),
        ("What is a catalyst?", "A catalyst speeds a reaction without being consumed."),
        ("What is evolution?", "Evolution is change in inherited traits over generations."),
    ]
    lines = []
    for user, assistant in pairs:
        lines.append(f"User: {user}\nAssistant: {assistant}\n")
        if user[0].isupper():
            low = user[0].lower() + user[1:]
            lines.append(f"User: {low}\nAssistant: {assistant}\n")
    return "".join(lines)


def build_multi_turn_conversations() -> str:
    conversations = [
        [
            ("Hi!", "Hello! What would you like to learn about today?"),
            ("Tell me about gravity.", "Gravity pulls objects with mass toward each other. Earth pulls you down."),
            ("Thanks, that helps.", "You are welcome! Ask anytime."),
        ],
        [
            ("Good morning.", "Good morning! Ready for a science question?"),
            ("What is energy?", "Energy is the ability to do work. It comes in many forms like motion and heat."),
            ("Cool.", "Glad that made sense!"),
        ],
        [
            ("I have a math question.", "Sure! What is it?"),
            ("What is 12 times 5?", "Twelve times five equals sixty."),
            ("Got it, thanks.", "Any time!"),
        ],
        [
            ("Explain friction simply.", "Friction is a force that slows things when surfaces rub together."),
            ("Like bike brakes?", "Exactly! Brake pads rub the wheel and create friction to stop you."),
            ("Nice.", "Happy to explain!"),
        ],
        [
            ("What is speed?", "Speed is distance divided by time."),
            ("Give an example.", "If you walk 6 kilometers in 2 hours, your speed is 3 km per hour."),
            ("That is clear.", "Great question!"),
        ],
        [
            ("What are atoms?", "Atoms are tiny building blocks of matter."),
            ("Do they combine?", "Yes! Atoms bond to form molecules like water, H2O."),
            ("Thank you.", "You are welcome!"),
        ],
        [
            ("Help with algebra.", "I can try! What equation do you have?"),
            ("x plus 7 equals 15.", "Subtract seven from both sides: x equals eight."),
            ("Perfect.", "Well done working through it!"),
        ],
        [
            ("What is photosynthesis?", "Plants use sunlight to make food and release oxygen."),
            ("Why does it matter?", "It produces oxygen we breathe and food for the food chain."),
            ("Makes sense.", "Science connects in many ways!"),
        ],
        [
            ("What is electricity?", "Electricity is moving electric charge, often through wires."),
            ("Is it dangerous?", "It can be! Always respect safety rules with outlets and storms."),
            ("OK thanks.", "Stay safe and curious!"),
        ],
        [
            ("Who was Isaac Newton?", "Isaac Newton was a scientist who described laws of motion and gravity."),
            ("When did he live?", "He lived from 1643 to 1727 in England."),
            ("Interesting.", "He changed physics forever!"),
        ],
        [
            ("What is a fraction?", "A fraction shows parts of a whole, like one half."),
            ("What is one half of 10?", "One half of ten is five."),
            ("Right.", "Fractions are everywhere in daily life!"),
        ],
        [
            ("Define density.", "Density is mass per unit volume."),
            ("Why do things float?", "Objects float when they are less dense than the fluid they displace."),
            ("Ah.", "Buoyancy follows density!"),
        ],
        [
            ("What is the periodic table?", "It lists all chemical elements by atomic number."),
            ("Give an example element.", "Oxygen is element number eight. We breathe O2."),
            ("Thanks.", "Chemistry is full of patterns!"),
        ],
        [
            ("How hot is the Sun?", "The Sun's surface is about 5500 degrees Celsius."),
            ("Wow.", "The core is even hotter, millions of degrees!"),
            ("That is wild.", "Stars are extreme!"),
        ],
        [
            ("What is probability?", "Probability measures how likely an event is, from 0 to 1."),
            ("Coin flip?", "A fair coin has one half chance of heads."),
            ("Easy enough.", "Probability helps with games and science!"),
        ],
        [
            ("Explain inertia.", "Inertia means objects resist changes in motion."),
            ("Like seat belts?", "Yes! Seat belts stop your body when the car stops suddenly."),
            ("Good example.", "Physics saves lives!"),
        ],
        [
            ("What is a cell?", "A cell is the basic unit of life."),
            ("Are bacteria cells?", "Yes, bacteria are single-celled organisms."),
            ("Neat.", "Life is built from cells!"),
        ],
        [
            ("Convert 1 km to meters.", "One kilometer equals one thousand meters."),
            ("And 2 hours to minutes?", "Two hours equals one hundred twenty minutes."),
            ("Helpful.", "Unit conversion is a key skill!"),
        ],
        [
            ("What is climate change?", "Climate change is long-term shift in global temperature and weather."),
            ("Main cause?", "Burning fossil fuels adds greenhouse gases like carbon dioxide."),
            ("I see.", "Understanding helps us act wisely."),
        ],
        [
            ("What is a right angle?", "A right angle is exactly ninety degrees."),
            ("Square corner?", "Yes, like the corner of a book."),
            ("Got it.", "Geometry is all around you!"),
        ],
    ]
    return "\n\n".join(_format_conversation(c) for c in conversations)


def expand_corpus(base: str, blocks: int, label: str) -> str:
    parts = [f"--- {label} block {i + 1} ---\n{base.strip()}" for i in range(blocks)]
    return "\n\n".join(parts)


def build_stem_corpus() -> str:
    multi = expand_corpus(build_multi_turn_conversations(), blocks=10, label="multi-turn")
    physics = expand_corpus(build_physics_articles(), blocks=4, label="physics")
    math_art = expand_corpus(build_math_articles(), blocks=4, label="math")
    chem = expand_corpus(build_chemistry_biology(), blocks=3, label="science")
    examples = expand_corpus(build_math_examples(), blocks=3, label="examples")
    qa = expand_corpus(build_stem_qa(), blocks=3, label="stem-qa")
    return (
        section("Physics", physics)
        + section("Math", math_art)
        + section("Chemistry and Biology", chem)
        + section("Math Examples", examples)
        + section("STEM Q&A", qa)
        + section("Multi-turn Chat", multi)
    )
