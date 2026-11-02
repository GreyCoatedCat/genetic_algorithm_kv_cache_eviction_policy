"""Tugas Needle-in-a-Haystack dengan banyak needle per sampel (mirip MQ-NIAH RULER).

Skornya recall: proporsi string jawaban yang muncul di output model. Dengan
beberapa needle, skor sampel bertingkat sehingga GA punya sinyal yang lebih halus
daripada benar/salah.
"""

import random
from dataclasses import dataclass

CITIES = [
    "Lima", "Oslo", "Cairo", "Hanoi", "Quito", "Dakar", "Riga", "Doha", "Kyiv", "Lagos",
    "Minsk", "Seoul", "Tunis", "Sofia", "Accra", "Amman", "Baku", "Bern", "Bogota", "Havana",
]
SUBJECTS = [
    "The grass", "The sky", "The sun", "The river", "The old road", "The market", "The harbor", "The forest",
    "The village", "The mountain", "The library", "The garden", "The winter wind", "The evening train",
]
PREDICATES = [
    "is green today.", "looks calm and quiet.", "was mentioned in the story.", "feels warm in the morning.",
    "stays the same as ever.", "is known to everyone here.", "changed a little over the years.",
    "appears on most of the maps.", "was busy all day long.", "is close to the town center.",
]


@dataclass
class Sample:
    input_ids: list[int]
    answers: list[str]

    def score(self, output: str) -> float:
        return sum(answer in output for answer in self.answers) / len(self.answers)


def make_sample(tokenizer, context_tokens: int, num_needles: int, rng: random.Random) -> Sample:
    cities = rng.sample(CITIES, num_needles)
    answers = [str(rng.randint(1_000_000, 9_999_999)) for _ in cities]
    needles = [f"The special magic number for {c} is: {a}." for c, a in zip(cities, answers)]

    # Ukuran filler dihitung dari rata-rata token per kalimat, bukan ditokenisasi satu per satu.
    def filler() -> str:
        return f"{rng.choice(SUBJECTS)} {rng.choice(PREDICATES)}"

    per_sentence = sum(len(tokenizer.encode(filler())) for _ in range(50)) / 50
    sentences = [filler() for _ in range(int(context_tokens / per_sentence))]
    for needle in needles:
        sentences.insert(rng.randint(0, len(sentences)), needle)

    question = f"What are the special magic numbers for {', '.join(cities)}?"
    prompt = (
        "Some special magic numbers are hidden in the text below. Memorize them.\n\n"
        + " ".join(sentences)
        + f"\n\nReply with only the numbers. {question}"
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    return Sample(tokenizer(text, add_special_tokens=False).input_ids, answers)


def make_samples(tokenizer, count: int, context_tokens: int, num_needles: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    return [make_sample(tokenizer, context_tokens, num_needles, rng) for _ in range(count)]
