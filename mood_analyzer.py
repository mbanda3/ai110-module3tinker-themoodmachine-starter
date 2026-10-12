# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

import re

# Emoticons/emojis are kept as tokens so they can act as mood signals.
TOKEN_PATTERN = re.compile(r":-?\)|:-?\(|[🌀-🫿☀-➿]|[\w']+")

# Enhancement: emoji/emoticon signals (weight added to the score).
EMOJI_SCORES = {
    ":)": 1, ":-)": 1, "😂": 1, "😍": 1, "🔥": 1, "🎉": 1, "😊": 1,
    ":(": -1, ":-(": -1, "😭": -1, "💀": -1, "😡": -1, "😞": -1,
}
NEGATORS = {"not", "never", "no", "dont", "cant", "wont", "isnt"}

class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        text = text.lower()
        text = re.sub(r"(\w){2,}", r"", text)  # "soooo" -> "soo"
        # Keep emoticons and emojis as their own tokens; drop other punctuation.
        return TOKEN_PATTERN.findall(text)

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        return self._analyze(text)[0]

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """Return (score, positive_hits, negative_hits) for the text."""
        tokens = self.preprocess(text)

        score = 0
        pos_hits: List[str] = []
        neg_hits: List[str] = []
        negate = False

        for i, token in enumerate(tokens):
            # Negation: flips the next sentiment word ("not happy").
            # "no cap" is slang for "for real", so it is not a negation.
            if token.replace("'", "") in NEGATORS:
                negate = not (token == "no" and tokens[i + 1:i + 2] == ["cap"])
                continue

            if token in EMOJI_SCORES:
                value = EMOJI_SCORES[token]
            elif token in self.positive_words:
                value = 1
            elif token in self.negative_words:
                value = -1
            else:
                negate = False
                continue

            if negate:
                value = -value
            score += value
            (pos_hits if value > 0 else neg_hits).append(token)
            negate = False

        return score, pos_hits, neg_hits

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        # TODO: Implement this method.
        #   1. Call self.score_text(text) to get the numeric score.
        #   2. Return "positive" if the score is above 0.
        #   3. Return "negative" if the score is below 0.
        #   4. Return "neutral" otherwise.
        score, pos_hits, neg_hits = self._analyze(text)

        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        elif pos_hits and neg_hits:
            return "mixed"  # signals of both kinds that cancel out
        return "neutral"


    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)
        score, pos_hits, neg_hits = self._analyze(text)

        return (
            f"Tokens = {tokens} | Score = {score} "
            f"(positive: {pos_hits or '[]'}, negative: {neg_hits or '[]'})"
        )
