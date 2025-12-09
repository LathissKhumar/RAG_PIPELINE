from typing import List, Dict
import re


def _split_paragraphs(text: str) -> List[str]:
    # split on two or more newlines
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_into_parts(text: str, max_chars: int) -> List[str]:
    paragraphs = _split_paragraphs(text)
    parts: List[str] = []
    buf = []
    buf_len = 0
    for p in paragraphs:
        if buf_len + len(p) + 2 <= max_chars:
            buf.append(p)
            buf_len += len(p) + 2
        else:
            if buf:
                parts.append("\n\n".join(buf))
            # paragraph too large for bucket: split by sentences
            if len(p) > max_chars:
                # naive sentence split by period followed by space
                sentences = re.split(r"(?<=[.!?])\s+", p)
                s_buf = []
                s_len = 0
                for s in sentences:
                    if s_len + len(s) + 1 <= max_chars:
                        s_buf.append(s)
                        s_len += len(s) + 1
                    else:
                        if s_buf:
                            parts.append(" ".join(s_buf))
                        # long single sentence fallback to hard split
                        if len(s) > max_chars:
                            for i in range(0, len(s), max_chars):
                                parts.append(s[i : i + max_chars])
                            s_buf = []
                            s_len = 0
                        else:
                            s_buf = [s]
                            s_len = len(s) + 1
                if s_buf:
                    parts.append(" ".join(s_buf))
                buf = []
                buf_len = 0
            else:
                buf = [p]
                buf_len = len(p) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def optimize_chunks(
    chunks: List[Dict],
    min_chars: int = 800,
    max_chars: int = 2500,
    overlap_chars: int = 200,
) -> List[Dict]:
    """Post-process a list of chunk dicts (with 'text' keys) to be RAG-friendly.

    Strategy:
    - Merge consecutive small chunks until at least `min_chars`.
    - Split chunks larger than `max_chars` at paragraph or sentence boundaries.
    - Add `overlap_chars` characters from the end of the previous chunk to the next chunk.
    """
    filtered = [c for c in chunks if c.get("text", "").strip()]
    merged: List[Dict] = []
    current = None

    for c in filtered:
        text = c.get("text", "").strip()
        if not text:
            continue

        if current is None:
            current = {"text": text, "meta": c.get("meta")}
            continue

        if len(current["text"]) < min_chars:
            # merge into current
            current["text"] = current["text"] + "\n\n" + text
            # merge meta into list form if possible
            try:
                if isinstance(current.get("meta"), list):
                    current["meta"].append(c.get("meta"))
                else:
                    current["meta"] = [current.get("meta"), c.get("meta")]
            except Exception:
                current["meta"] = {"merged": True}
        else:
            # current is big enough, push and start new
            merged.append(current)
            current = {"text": text, "meta": c.get("meta")}

    if current is not None:
        merged.append(current)

    # now split large chunks
    final: List[Dict] = []
    for item in merged:
        t = item["text"]
        if len(t) <= max_chars:
            final.append(item)
            continue
        parts = _split_into_parts(t, max_chars)
        for p in parts:
            final.append({"text": p, "meta": item.get("meta")})

    # apply overlap
    if overlap_chars > 0 and final:
        out: List[Dict] = []
        prev = None
        for it in final:
            txt = it["text"]
            if prev is None:
                out.append(it)
                prev = it
                continue
            # take last overlap_chars from prev
            prev_tail = prev["text"][-overlap_chars:]
            new_text = prev_tail + "\n\n" + txt
            out.append({"text": new_text, "meta": it.get("meta")})
            prev = it
        final = out

    return final
