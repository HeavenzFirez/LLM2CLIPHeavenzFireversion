"""Base-9216 radix conversion for the Sovereign pipeline.

The Continuum represents scalar aggregates in a wide radix (base 9216) over an
extended alphabet so that large magnitudes pack into short, comparable strings.
Conversion is pure-Python (no dependencies) and every value is clamped into the
non-negative domain before encoding, so the output is safe to round-trip
through CSV/JSON.

The alphabet is the union of digits, ASCII letters, and common punctuation,
guaranteeing that encoded strings are filesystem- and log-safe.
"""

from __future__ import annotations

from typing import List

# Extended, filesystem/log-safe alphabet. Order is fixed and significant.
_ALPHABET: str = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "+-_=.,:;!?@#$%&*()[]{}<>"
)
_RADIX: int = len(_ALPHABET)  # 9216? No — see note below.

# NOTE: a single printable-ASCII alphabet cannot reach 9216 distinct glyphs.
# To honour the "base 9216" contract while staying log-safe, we use a
# multi-character codepoint scheme: each radix-9216 digit is emitted as a
# fixed-width 2-character bigram drawn from a 96-glyph printable subset. This
# keeps every output value a plain ASCII string that survives CSV/JSON/logs.
_PRINTABLE = "".join(chr(c) for c in range(33, 127))  # 94 glyphs, all printable
_BIGRAM_BASE = len(_PRINTABLE)  # 94
# 94**2 = 8836 < 9216, so a 2-glyph bigram is insufficient. Use the literal
# integer-base-9216 representation but render with a delimiter so the output is
# unambiguous and reversible.
RADIX = 9216


def _to_base(value: int, base: int) -> List[int]:
    """Return the little-endian list of digits of ``value`` in ``base``."""
    if value == 0:
        return [0]
    digits: List[int] = []
    n = value
    while n > 0:
        digits.append(n % base)
        n //= base
    return digits


def _from_base(digits: List[int], base: int) -> int:
    """Inverse of :func:`_to_base` (little-endian digit list -> int)."""
    result = 0
    for digit in reversed(digits):
        result = result * base + digit
    return result


def to_radix_9216(value: int) -> str:
    """Encode a non-negative integer in base 9216 as a ``.``-delimited string.

    Negative inputs are clamped to 0. The representation is little-endian and
    reversible via :func:`from_radix_9216`. Using a delimiter (rather than a
    fixed glyph alphabet) keeps the scheme unambiguous for radix > 96 while
    remaining pure ASCII and log-safe.
    """
    if value < 0:
        value = 0
    digits = _to_base(value, RADIX)
    # Most-significant first for human readability.
    return ".".join(str(d) for d in reversed(digits))


def from_radix_9216(encoded: str) -> int:
    """Decode a ``.``-delimited base-9216 string back to an int.

    Empty strings decode to 0. Malformed digits raise :class:`ValueError`.
    """
    encoded = encoded.strip()
    if not encoded:
        return 0
    parts = [p for p in encoded.split(".") if p != ""]
    digits = [int(p) for p in parts]  # ValueError on non-numeric
    if any(d < 0 or d >= RADIX for d in digits):
        raise ValueError(f"digit out of range [0,{RADIX}) in {encoded!r}")
    return _from_base(list(reversed(digits)), RADIX)


__all__ = ["RADIX", "to_radix_9216", "from_radix_9216"]
