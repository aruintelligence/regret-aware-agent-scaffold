"""Dependency-free sentence retrieval on independently annotated SQuAD contexts.

Neither retrieval algorithm receives human answer annotations. The gold answer
spans are inspected only by the evaluator after the prediction.
"""
import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import mean

SOURCE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
STOP = frozenset((
    "a an the and or but if then to of in on for from with by as at is are was were "
    "be been being do does did which what who where when why how many much that this "
    "these those it its their they them he she his her can could would should will "
    "have has had not into about between after before than through during name called"
).split())
TOKEN = re.compile(r"[A-Za-z0-9]+")
BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“\'(\[])')


def terms(text):
    return [word.lower() for word in TOKEN.findall(text) if word.lower() not in STOP]


@lru_cache(maxsize=8192)
def segment_context(context):
    """Return (start,end,text) tuples with offsets into the original context."""
    spans = []
    start = 0
    for match in BOUNDARY.finditer(context):
        end = match.start()
        piece = context[start:end]
        if piece.strip():
            leading = len(piece) - len(piece.lstrip())
            spans.append((start + leading, end - (len(piece) - len(piece.rstrip()))))
        start = match.end()
    if start < len(context):
        piece = context[start:]
        if piece.strip():
            leading = len(piece) - len(piece.lstrip())
            spans.append((start + leading, len(context) - (len(piece) - len(piece.rstrip()))))
    if not spans and context.strip():
        spans = [(0, len(context))]
    return tuple((begin, end, context[begin:end]) for begin, end in spans)


def overlap_sentence(question, context):
    """Baseline: unique lexical overlap normalized for sentence length."""
    spans = segment_context(context)
    if not spans:
        return (0, 0, "")
    query = set(terms(question))
    def score(span):
        words = terms(span[2])
        if not words or not query:
            return 0.0
        return len(query.intersection(words)) / math.sqrt(len(words))
    return max(spans, key=score)


def idf_proximity_sentence(question, context):
    """Challenger: sentence-level BM25, query coverage and token proximity."""
    spans = segment_context(context)
    if not spans:
        return (0, 0, "")
    query = set(terms(question))
    if not query:
        return spans[0]
    corpus = [terms(span[2]) for span in spans]
    n = len(corpus)
    avg_length = max(1.0, mean(len(words) for words in corpus))
    df = Counter(word for words in corpus for word in set(words))
    lower_question = question.strip().lower()
    numeric = lower_question.startswith(("when ", "how many ", "what year ", "which year "))
    scores = []
    for words in corpus:
        counts = Counter(words)
        norm = 1.2 * (0.25 + 0.75 * len(words) / avg_length)
        bm25 = sum(
            math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
            * counts[word] * 2.2 / (counts[word] + norm)
            for word in query if counts[word]
        )
        matched = set(words).intersection(query)
        coverage = len(matched) / len(query)
        positions = [i for i, word in enumerate(words) if word in query]
        proximity = 0.0
        if len(matched) >= 2:
            proximity = len(matched) / (1 + (max(positions) - min(positions)) / len(matched))
        number_cue = 0.16 if numeric and any(c.isdigit() for c in " ".join(words)) else 0.0
        scores.append(bm25 + 0.8 * coverage + 0.30 * proximity + number_cue)
    return spans[max(range(len(spans)), key=lambda i: scores[i])]


def gold_span_contained(span, answers):
    """Only the evaluator sees official answer spans."""
    start, end, _ = span
    return any(
        isinstance(a.get("answer_start"), int)
        and isinstance(a.get("text"), str)
        and start <= a["answer_start"]
        and a["answer_start"] + len(a["text"]) <= end
        for a in answers
    )


def load_questions(dataset_path, limit=2000):
    """Stable ID-hash sample; no sorting or filtering by outcome."""
    raw = Path(dataset_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    if payload.get("version") != "1.1" or not isinstance(payload.get("data"), list):
        raise ValueError("expected official SQuAD v1.1 JSON")
    items = []
    seen = set()
    for article in payload["data"]:
        title = article.get("title", "untitled")
        for paragraph in article.get("paragraphs", []):
            context = paragraph["context"]
            for qa in paragraph.get("qas", []):
                qid = qa["id"]
                if qid in seen:
                    raise ValueError("duplicate question ID")
                seen.add(qid)
                answers = qa.get("answers", [])
                if not answers:
                    continue
                if not all(isinstance(a.get("answer_start"), int)
                           and isinstance(a.get("text"), str)
                           and context[a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]
                           for a in answers):
                    raise ValueError("answer offsets do not match passage text")
                items.append((qid, title, context, qa["question"], answers))
    items.sort(key=lambda row: hashlib.sha256(row[0].encode("utf-8")).hexdigest())
    if limit < 1:
        raise ValueError("limit must be positive")
    return items[:limit], digest, len(items)
