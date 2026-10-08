"""Form→meaning residual transparency pipeline."""

from __future__ import annotations  # postponed evaluation of type hints

__all__ = [  # names this package advertises
    "load_word_csv",  # lexicon CSV loader (defined in opacity.lexicon)
    "cross_validated_opacity",  # held-out transparency scorer (defined in opacity.scores)
]
