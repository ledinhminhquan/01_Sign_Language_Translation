"""The synthetic sign vocabulary: GLOSS -> spoken-text lexicon.

Each gloss maps to its spoken-English word(s). The text deliberately differs from the gloss token
(lowercase, multi-word like "thank you", pronoun mapping ME->"i") so that the gloss->text
translation stage is non-trivial vs an identity baseline. The first ``vocab_size`` entries form
the active vocabulary; ordering is fixed for reproducibility.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# (GLOSS, spoken text) — ASL-style glossing into English words.
GLOSS_TEXT: List[Tuple[str, str]] = [
    ("HELLO", "hello"), ("ME", "i"), ("YOU", "you"), ("NAME", "name"),
    ("THANK-YOU", "thank you"), ("YES", "yes"), ("NO", "no"), ("GOOD", "good"),
    ("BAD", "bad"), ("LEARN", "learn"), ("SIGN", "sign"), ("WANT", "want"),
    ("HELP", "help"), ("EAT", "eat"), ("DRINK", "drink"), ("HAPPY", "happy"),
    ("FRIEND", "friend"), ("TEACHER", "teacher"), ("SCHOOL", "school"), ("DAY", "day"),
    ("TODAY", "today"), ("TOMORROW", "tomorrow"), ("NOW", "now"), ("LOVE", "love"),
    ("FAMILY", "family"), ("HOME", "home"), ("WORK", "work"), ("BOOK", "book"),
    ("WATER", "water"), ("FOOD", "food"), ("MORE", "more"), ("FINISH", "finish"),
    ("UNDERSTAND", "understand"), ("QUESTION", "question"), ("PLEASE", "please"),
    ("SORRY", "sorry"), ("WELCOME", "welcome"), ("SEE", "see"), ("GO", "go"),
    ("COME", "come"), ("KNOW", "know"), ("TIME", "time"), ("HOW", "how"),
    ("WHAT", "what"), ("WHERE", "where"), ("WHO", "who"), ("WHY", "why"),
    ("GREAT", "great"), ("SLOW", "slow"), ("FAST", "fast"),
]


def vocab(vocab_size: int) -> List[str]:
    return [g for g, _ in GLOSS_TEXT[:vocab_size]]


def gloss_to_text(vocab_size: int) -> Dict[str, str]:
    return {g: t for g, t in GLOSS_TEXT[:vocab_size]}


def translate_glosses(glosses: List[str], vocab_size: int) -> str:
    """Lexicon translation: gloss sequence -> spoken text (the offline dictionary translator)."""
    lex = gloss_to_text(max(vocab_size, len(GLOSS_TEXT)))
    return " ".join(lex.get(g, g.lower()) for g in glosses)


__all__ = ["GLOSS_TEXT", "vocab", "gloss_to_text", "translate_glosses"]
