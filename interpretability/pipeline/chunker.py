"""Recursive char-based text splitter for the RAG variants.

Splits cleaned body text into overlapping chunks suitable for sentence-
embedding retrieval. The splitter is char-based (not token-based) to avoid a
tokenizer dependency at chunk time — chunks are sized so that bge-small-en-v1.5
(max 512 tokens) can encode them comfortably without truncation.

Algorithm:
    1. Split on paragraph boundaries (\\n\\n).
    2. If a piece exceeds ``size``, recursively split on ". " and finally on
       whitespace so no leaf piece is larger than ``size``.
    3. Greedily pack pieces into chunks. Emit a chunk when accumulated length
       reaches ``size``; carry the last ``overlap`` chars into the next chunk.
    4. Drop the trailing chunk if its length is below ``min_size`` and it is
       not the only chunk produced (a tiny tail adds noise to retrieval).
"""

from __future__ import annotations

import re

_PARA = re.compile(r"\n\s*\n+")
_SENT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


def _split_to_pieces(text: str, size: int) -> list[str]:
    pieces: list[str] = []
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            pieces.append(para)
            continue
        for sent in _SENT.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= size:
                pieces.append(sent)
                continue
            words = _WS.split(sent)
            cur: list[str] = []
            cur_len = 0
            for w in words:
                add = len(w) + (1 if cur else 0)
                if cur_len + add > size and cur:
                    pieces.append(" ".join(cur))
                    cur = [w]
                    cur_len = len(w)
                else:
                    cur.append(w)
                    cur_len += add
            if cur:
                pieces.append(" ".join(cur))
    return pieces


def chunk_text(
    text: str,
    *,
    size: int = 800,
    overlap: int = 200,
    min_size: int = 100,
) -> list[str]:
    """Split ``text`` into overlapping chunks of approximately ``size`` chars."""
    if not text:
        return []
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap={overlap} must satisfy 0 <= overlap < size={size}")

    pieces = _split_to_pieces(text, size)
    if not pieces:
        return []

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if not cur:
            cur = piece
            continue
        if len(cur) + 1 + len(piece) <= size:
            cur = f"{cur} {piece}"
            continue
        chunks.append(cur)
        carry = cur[-overlap:] if overlap else ""
        cur = f"{carry} {piece}".strip() if carry else piece
    if cur:
        chunks.append(cur)

    if len(chunks) > 1 and len(chunks[-1]) < min_size:
        chunks.pop()

    return chunks
