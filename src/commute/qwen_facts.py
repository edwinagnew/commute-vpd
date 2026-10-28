"""Anchor 2 of E45 (plan reports/plans/2026-10-28_plan_qwen-q2.md, section 3): question-answer pairs about invented
people, with known facts mixed in. Pure and seeded: the same seed gives the same facts, questions and order.

200 invented people (names built from a syllable list, each drawn once) x 5 relations (home town, employer, profession,
birth year, instrument) = 1,000 invented facts; 100 country capitals and 100 element symbols = 200 known facts. Each
relation has four question templates: templates 0-2 are trained, template 3 is the held-out reading. A question is the
template filled with the subject, then FORMAT; the answer is "The answer is X." Seed 1 is a second draw of names and
attributes (Fable, 2026-10-28); the known facts are the same for every seed.

Grading: the text after the last "The answer is" on its line, with "*" removed, surrounding spaces and a final "."
stripped, accents removed, compared case-insensitively to the answer.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass

N_PEOPLE, TRAIN_TEMPLATES, HELD_TEMPLATE = 200, (0, 1, 2), 3
FORMAT = "Reply with one line of the form: The answer is X."

SYLLABLES = ("ka", "lo", "mer", "vin", "tas", "rel", "dor", "py", "sen", "gal", "ost", "bri", "nu", "fel", "ham",
             "zor", "qui", "ber", "tal", "mon", "ves", "ril", "dax", "po", "lun", "sar", "eth", "kor", "vel", "mi")
TOWN_ENDS = ("ford", "by", "wick", "stead", "mouth", "ton", "dale", "holm")
FIRM_ENDS = ("Works", "Labs", "Group", "Industries", "Partners", "Systems")
PROFESSIONS = ("baker", "surveyor", "pharmacist", "carpenter", "librarian", "geologist", "electrician", "translator",
               "architect", "veterinarian", "plumber", "journalist", "optician", "brewer", "cartographer", "dentist",
               "glassblower", "locksmith", "meteorologist", "beekeeper", "tailor", "welder", "astronomer", "potter",
               "notary", "chemist", "florist", "machinist", "auditor", "botanist")
INSTRUMENTS = ("violin", "cello", "oboe", "clarinet", "trumpet", "harp", "flute", "bassoon", "trombone", "viola",
               "banjo", "accordion", "mandolin", "tuba", "harpsichord", "saxophone", "piccolo", "marimba", "sitar",
               "bagpipes")
CAPITALS = (
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("India", "New Delhi"), ("Russia", "Moscow"), ("Canada", "Ottawa"),
    ("Australia", "Canberra"), ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"), ("Mexico", "Mexico City"),
    ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("South Korea", "Seoul"), ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"), ("Turkey", "Ankara"), ("Greece", "Athens"),
    ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Hungary", "Budapest"), ("Sweden", "Stockholm"), ("Norway", "Oslo"),
    ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Ireland", "Dublin"), ("Netherlands", "Amsterdam"),
    ("Belgium", "Brussels"), ("Switzerland", "Bern"), ("Czech Republic", "Prague"), ("Romania", "Bucharest"),
    ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Malta", "Valletta"), ("Belarus", "Minsk"),
    ("Lithuania", "Vilnius"), ("Latvia", "Riga"), ("Estonia", "Tallinn"), ("Iceland", "Reykjavik"), ("Peru", "Lima"),
    ("Chile", "Santiago"), ("Colombia", "Bogota"), ("Venezuela", "Caracas"), ("Ecuador", "Quito"), ("Cuba", "Havana"),
    ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion"), ("Slovenia", "Ljubljana"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
    ("Syria", "Damascus"), ("Jordan", "Amman"), ("Lebanon", "Beirut"), ("Saudi Arabia", "Riyadh"), ("Qatar", "Doha"),
    ("Afghanistan", "Kabul"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Nepal", "Kathmandu"),
    ("Sri Lanka", "Colombo"), ("Malaysia", "Kuala Lumpur"), ("Singapore", "Singapore"), ("Mongolia", "Ulaanbaatar"),
    ("Kazakhstan", "Astana"), ("Uzbekistan", "Tashkent"), ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"),
    ("Azerbaijan", "Baku"), ("Cyprus", "Nicosia"), ("Ethiopia", "Addis Ababa"), ("Ghana", "Accra"),
    ("Senegal", "Dakar"), ("Morocco", "Rabat"), ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Libya", "Tripoli"),
    ("Sudan", "Khartoum"), ("Uganda", "Kampala"), ("Rwanda", "Kigali"), ("Zimbabwe", "Harare"), ("Zambia", "Lusaka"),
    ("Angola", "Luanda"), ("Mozambique", "Maputo"), ("Madagascar", "Antananarivo"), ("New Zealand", "Wellington"),
    ("Fiji", "Suva"), ("Jamaica", "Kingston"), ("Panama", "Panama City"), ("Costa Rica", "San Jose"),
    ("Guatemala", "Guatemala City"), ("Honduras", "Tegucigalpa"), ("Nicaragua", "Managua"), ("Slovakia", "Bratislava"),
)
_SYMBOLS = (  # atomic numbers 1-100
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo "
    "Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl "
    "Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm")
_NAMES = (
    "Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon "
    "Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel "
    "Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium "
    "Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium "
    "Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium "
    "Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury Thallium "
    "Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium "
    "Americium Curium Berkelium Californium Einsteinium Fermium")
ELEMENTS, ELEMENT_NAMES = tuple(_SYMBOLS.split()), tuple(_NAMES.split())
TEMPLATES = {
    "hometown": ("Which town is {x} from?", "Where does {x} come from? Name the town.", "What is the home town of {x}?",
                 "In which town did {x} grow up?"),
    "employer": ("Which company does {x} work for?", "Who employs {x}?", "What is the name of {x}'s employer?",
                 "At which company is {x} employed?"),
    "profession": ("What does {x} do for a living?", "What is {x}'s profession?", "What job does {x} have?",
                   "What is the occupation of {x}?"),
    "birth_year": ("In what year was {x} born?", "What is {x}'s year of birth?", "When was {x} born? Give the year.",
                   "Which year was {x} born in?"),
    "instrument": ("Which instrument does {x} play?", "What musical instrument does {x} play?",
                   "What instrument is {x} known for playing?", "On which instrument does {x} perform?"),
    "capital": ("What is the capital of {x}?", "Which city is the capital of {x}?", "Name the capital city of {x}.",
                "What city serves as the capital of {x}?"),
    "symbol": ("What is the chemical symbol of {x}?", "Which symbol stands for the element {x}?",
               "What is the element symbol for {x}?", "How is {x} written as a chemical symbol?"),
}
_ANSWER = re.compile(r"The answer is([^\n]*)")


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    answer: str
    known: bool


@dataclass(frozen=True)
class QA:
    question: str  # the user turn: the filled template, a blank line, FORMAT
    answer: str  # the response: "The answer is X."
    fact: Fact
    template: int


def _word(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(n)).capitalize()


def invented(seed: int, n_people: int = N_PEOPLE) -> list[Fact]:
    """n_people invented people, five facts each, in person order; names are distinct within a draw."""
    rng, names, facts = random.Random(f"qwen_facts-{seed}"), set(), []
    while len(names) < n_people:
        name = f"{_word(rng, 2)} {_word(rng, 3)}"
        if name in names:
            continue
        names.add(name)
        attrs = {"hometown": _word(rng, 2) + rng.choice(TOWN_ENDS),
                 "employer": f"{_word(rng, 2)} {rng.choice(FIRM_ENDS)}",
                 "profession": rng.choice(PROFESSIONS), "birth_year": str(rng.randint(1930, 1999)),
                 "instrument": rng.choice(INSTRUMENTS)}
        facts += [Fact(name, rel, ans, known=False) for rel, ans in attrs.items()]
    return facts


def known() -> list[Fact]:
    return ([Fact(c, "capital", a, known=True) for c, a in CAPITALS]
            + [Fact(n, "symbol", s, known=True) for n, s in zip(ELEMENT_NAMES, ELEMENTS, strict=True)])


def qa(fact: Fact, template: int) -> QA:
    q = TEMPLATES[fact.relation][template].format(x=fact.subject)
    return QA(f"{q}\n\n{FORMAT}", f"The answer is {fact.answer}.", fact, template)


def train_set(seed: int) -> list[QA]:
    """Every invented and known fact under templates 0-2 (3,600 pairs at the default sizes), unshuffled."""
    return [qa(f, t) for f in invented(seed) + known() for t in TRAIN_TEMPLATES]


def held_set(seed: int) -> list[QA]:
    """Every invented and known fact under the held-out template 3."""
    return [qa(f, HELD_TEMPLATE) for f in invented(seed) + known()]


def norm(s: str) -> str:
    """Accents, "*", a final "." and extra spaces removed; case folded ("**Brasília.**" reads "brasilia")."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return " ".join(plain.replace("*", "").strip().rstrip(".").split()).casefold()


def correct(response: str, answer: str) -> bool:
    hits = _ANSWER.findall(response)
    return bool(hits) and norm(hits[-1]) == norm(answer)
