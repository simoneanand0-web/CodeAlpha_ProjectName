#!/usr/bin/env python3
"""Create an aggregate-only sentiment and emotion report from a review CSV.

Expected columns are a review text field (review/text/sentence/comment/content)
and, optionally, a sentiment label or rating. The implementation uses a small,
transparent word lexicon and standard-library CSV handling; no raw review text
is copied into the generated report.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
import re
from collections import Counter
from pathlib import Path


POSITIVE = set("""
amazing awesome best beautiful brilliant comfortable cool delight delightful
easy excellent exceptional fantastic favorite flawless fresh good great happy
helpful incredible love loved lovely nice outstanding perfect pleased positive
recommend reliable satisfied smooth solid superb terrific thank thanks useful
value wonderful works well
""".split())

NEGATIVE = set("""
awful bad broke broken disappointing disappointed dislike displeased fail failed
garbage hate hated horrible issue junk lousy low poor problem refund regret
return returned sad slow terrible unfortunately waste worst worthless wrong
""".split())

EMOTIONS = {
    "joy": set("happy love loved delighted pleased excited glad joy wonderful smile".split()),
    "trust": set("reliable recommend dependable honest secure confident trust".split()),
    "anger": set("angry hate hated furious frustrating annoying annoyed mad".split()),
    "sadness": set("sad disappointed disappointing unhappy regret sorry".split()),
    "fear": set("afraid worried fear concern concerned unsafe risk".split()),
    "disgust": set("awful disgusting gross nasty horrible terrible".split()),
    "anticipation": set("expect expecting hope hoped looking soon await".split()),
    "surprise": set("surprised surprise unexpectedly amazing shocking".split()),
}

NEGATORS = {"not", "no", "never", "hardly", "isnt", "wasnt", "dont", "didnt", "cant", "wont"}
BOOSTERS = {"very": 1.45, "really": 1.35, "extremely": 1.7, "totally": 1.45, "highly": 1.4}
DIMINISHERS = {"slightly": 0.65, "somewhat": 0.7, "barely": 0.6, "almost": 0.75}
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
STOP_WORDS = set("""
a an and are as at be been being but by for from had has have he her hers him his
i if in into is it its me my of on or our ours she so than that the their them
they this to was we were what when which who will with you your yours also just
very really product phone item use used get got one would could can dont didnt
not
""".split())
TEXT_COLUMNS = ("review", "text", "sentence", "comment", "content", "review_text")
LABEL_COLUMNS = ("label", "sentiment", "score", "rating", "polarity")


def normalize_label(value: str | None, column_name: str | None = None) -> str | None:
    if value is None:
        return None
    value = value.strip().casefold()
    if column_name and column_name.casefold() in {"rating", "ratings", "stars", "star_rating"}:
        if value in {"4", "5"}:
            return "positive"
        if value == "3":
            return "neutral"
        if value in {"1", "2"}:
            return "negative"
    if value in {"positive", "pos", "+", "1", "5", "4", "good"}:
        return "positive"
    if value in {"negative", "neg", "-", "0", "2", "bad"}:
        return "negative"
    if value in {"neutral", "neu", "3", "mixed"}:
        return "neutral"
    return None


def infer_columns(fieldnames: list[str] | None, text_column: str | None, label_column: str | None) -> tuple[str, str | None]:
    names = fieldnames or []
    lowered = {name.casefold(): name for name in names}
    selected_text = text_column or next((lowered[name] for name in TEXT_COLUMNS if name in lowered), None)
    selected_label = label_column or next((lowered[name] for name in LABEL_COLUMNS if name in lowered), None)
    if selected_text is None:
        if len(names) == 2:
            selected_text = names[0]
            selected_label = selected_label or names[1]
        else:
            raise ValueError("Could not identify a review-text column. Name it review, text, sentence, comment, or content.")
    if selected_text not in names:
        raise ValueError(f"Text column {selected_text!r} was not found. Available columns: {', '.join(names)}")
    if selected_label and selected_label not in names:
        raise ValueError(f"Label column {selected_label!r} was not found. Available columns: {', '.join(names)}")
    return selected_text, selected_label


def read_rows(path: Path, text_column: str | None, label_column: str | None) -> list[dict[str, str | None]]:
    delimiter = "\t" if path.suffix.casefold() in {".tsv", ".txt"} else ","
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("The input file needs a header row naming the review text column.")
        text_key, label_key = infer_columns(reader.fieldnames, text_column, label_column)
        rows: list[dict[str, str | None]] = []
        for line_number, row in enumerate(reader, start=2):
            review = (row.get(text_key) or "").strip()
            if not review:
                continue
            rows.append({"text": review, "label": normalize_label(row.get(label_key), label_key) if label_key else None})
    if not rows:
        raise ValueError("No non-empty reviews were found in the input file.")
    return rows


def classify(text: str) -> tuple[str, float, Counter[str]]:
    tokens = [token.casefold().replace("'", "") for token in TOKEN_RE.findall(text)]
    raw_score = 0.0
    emotions: Counter[str] = Counter()
    for index, token in enumerate(tokens):
        for emotion, words in EMOTIONS.items():
            if token in words:
                emotions[emotion] += 1
        polarity = 1.0 if token in POSITIVE else -1.0 if token in NEGATIVE else 0.0
        if polarity == 0:
            continue
        previous = tokens[max(0, index - 3):index]
        if any(word in NEGATORS for word in previous):
            polarity *= -1
        if previous:
            polarity *= BOOSTERS.get(previous[-1], DIMINISHERS.get(previous[-1], 1.0))
        if "but" in tokens[:index]:
            polarity *= 1.15
        raw_score += polarity
    score = math.tanh(raw_score / math.sqrt(max(len(tokens), 1)))
    if text.count("!") >= 2:
        score = max(-1.0, min(1.0, score * 1.08))
    sentiment = "positive" if score >= 0.10 else "negative" if score <= -0.10 else "neutral"
    return sentiment, score, emotions


def model_features(text: str) -> Counter[str]:
    tokens = [token.casefold().replace("'", "") for token in TOKEN_RE.findall(text)]
    features: Counter[str] = Counter()
    for index, token in enumerate(tokens):
        previous = tokens[max(0, index - 3):index]
        if token in STOP_WORDS or len(token) < 2:
            continue
        if any(word in NEGATORS for word in previous):
            features[f"neg_{token}"] += 1
        else:
            features[token] += 1
    return features


def train_naive_bayes(training_rows: list[dict[str, str | None]]) -> tuple[dict[str, Counter[str]], dict[str, int], set[str]]:
    word_counts = {"positive": Counter(), "negative": Counter()}
    document_counts = {"positive": 0, "negative": 0}
    for row in training_rows:
        label = row["label"]
        if label not in word_counts:
            continue
        document_counts[label] += 1
        word_counts[label].update(model_features(str(row["text"])))
    vocabulary = set(word_counts["positive"]) | set(word_counts["negative"])
    return word_counts, document_counts, vocabulary


def predict_naive_bayes(text: str, model: tuple[dict[str, Counter[str]], dict[str, int], set[str]]) -> tuple[str, float]:
    word_counts, document_counts, vocabulary = model
    if not vocabulary or not document_counts["positive"] or not document_counts["negative"]:
        return "neutral", 0.0
    total_positive = sum(word_counts["positive"].values())
    total_negative = sum(word_counts["negative"].values())
    vocabulary_size = max(len(vocabulary), 1)
    log_odds = math.log(document_counts["positive"] / document_counts["negative"])
    for word, count in model_features(text).items():
        positive_probability = (word_counts["positive"][word] + 1) / (total_positive + vocabulary_size)
        negative_probability = (word_counts["negative"][word] + 1) / (total_negative + vocabulary_size)
        log_odds += count * math.log(positive_probability / negative_probability)
    bounded_odds = max(-40.0, min(40.0, log_odds))
    probability_positive = 1.0 / (1.0 + math.exp(-bounded_odds))
    signed_score = 2.0 * probability_positive - 1.0
    if 0.42 <= probability_positive <= 0.58:
        return "neutral", signed_score
    return ("positive" if probability_positive > 0.58 else "negative"), signed_score


def cross_validated_predictions(rows: list[dict[str, str | None]], folds: int = 5) -> list[tuple[str, float]]:
    grouped = {"positive": [], "negative": []}
    for index, row in enumerate(rows):
        if row["label"] in grouped:
            grouped[str(row["label"])].append(index)
    assignments: dict[int, int] = {}
    rng = random.Random(20261120)
    for label, indices in grouped.items():
        rng.shuffle(indices)
        # Stratified, seeded assignment keeps both classes represented in each fold.
        for offset, index in enumerate(indices):
            assignments[index] = offset % folds
    predictions: list[tuple[str, float] | None] = [None] * len(rows)
    for fold in range(folds):
        training = [row for index, row in enumerate(rows) if assignments.get(index) != fold]
        model = train_naive_bayes(training)
        for index, row in enumerate(rows):
            if assignments.get(index) == fold:
                predictions[index] = predict_naive_bayes(str(row["text"]), model)
    return [prediction if prediction is not None else classify(str(row["text"]))[:2]
            for prediction, row in zip(predictions, rows)]


def top_terms(rows: list[dict[str, object]], sentiment: str, limit: int = 8) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row["sentiment"] == sentiment:
            counts.update(token.casefold().replace("'", "") for token in TOKEN_RE.findall(str(row["text"]))
                          if token.casefold() not in STOP_WORDS and len(token) > 2)
    return counts.most_common(limit)


def svg_bars(title: str, values: list[tuple[str, int]], color: str) -> str:
    width, row_height, left, bar_width = 760, 36, 210, 410
    height = max(100, 45 + row_height * len(values))
    maximum = max((value for _, value in values), default=1) or 1
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<title>{html.escape(title)}</title>']
    for index, (label, value) in enumerate(values):
        y = 34 + index * row_height
        bar = value / maximum * bar_width
        parts.append(f'<text class="label" x="{left-12}" y="{y+17}" text-anchor="end">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y+3}" width="{bar:.1f}" height="20" rx="5" fill="{color}"/>')
        parts.append(f'<text class="value" x="{left+bar_width+12}" y="{y+18}">{value:,}</text>')
    parts.append("</svg>")
    return "".join(parts)


def word_table(values: list[tuple[str, int]]) -> str:
    if not values:
        return "<p class='muted'>No matching words were found.</p>"
    return "<table><thead><tr><th>Term</th><th>Mentions</th></tr></thead><tbody>" + "".join(
        f"<tr><td>{html.escape(word)}</td><td>{count:,}</td></tr>" for word, count in values
    ) + "</tbody></table>"


def build_report(input_path: Path, output_path: Path, source_name: str,
                 text_column: str | None = None, label_column: str | None = None) -> dict[str, object]:
    source_rows = read_rows(input_path, text_column, label_column)
    analyzed: list[dict[str, object]] = []
    for row in source_rows:
        sentiment, score, emotions = classify(str(row["text"]))
        analyzed.append({"text": row["text"], "label": row["label"], "sentiment": sentiment,
                         "score": score, "emotions": emotions})
    has_binary_labels = any(row["label"] in {"positive", "negative"} for row in source_rows)
    if has_binary_labels:
        predictions = cross_validated_predictions(source_rows)
        for row, (sentiment, score) in zip(analyzed, predictions):
            row["sentiment"], row["score"] = sentiment, score

    sentiment_counts = Counter(str(row["sentiment"]) for row in analyzed)
    sentiment_order = [(name, sentiment_counts[name]) for name in ("positive", "neutral", "negative")]
    emotion_counts = Counter()
    for row in analyzed:
        for emotion in row["emotions"]:
            emotion_counts[emotion] += 1
    emotion_order = [(name, emotion_counts[name]) for name in
                     ("joy", "trust", "anger", "sadness", "fear", "disgust", "anticipation", "surprise")]
    emotion_order = sorted(emotion_order, key=lambda item: (-item[1], item[0]))
    average_score = sum(float(row["score"]) for row in analyzed) / len(analyzed)
    labelled = [row for row in analyzed if row["label"] is not None]
    accuracy = (sum(row["sentiment"] == row["label"] for row in labelled) / len(labelled)) if labelled else None
    decisive = [row for row in labelled if row["sentiment"] != "neutral"]
    decisive_accuracy = (sum(row["sentiment"] == row["label"] for row in decisive) / len(decisive)) if decisive else None
    neutral_gold = any(row["label"] == "neutral" for row in labelled)
    top_positive = top_terms(analyzed, "positive")
    top_negative = top_terms(analyzed, "negative")

    if labelled and not neutral_gold:
        evaluation_note = ("Predictions use stratified 5-fold out-of-fold Naive Bayes classification with a negation-aware bag of words; "
                          "neutral is reserved for low-confidence scores. The supplied labels contain positive/negative examples only, "
                          "so neutral predictions cannot be validated against a neutral ground-truth class.")
    elif labelled:
        evaluation_note = "Accuracy compares the lexicon prediction with the supplied labels; neutral is included when present."
    else:
        evaluation_note = ("No labelled target column was found, so accuracy could not be measured. Sentiment labels use the built-in "
                          "positive/negative word lexicon and a neutral score threshold.")
    accuracy_text = f"{accuracy:.1%}" if accuracy is not None else "Not available"
    decisive_accuracy_text = f"{decisive_accuracy:.1%}" if decisive_accuracy is not None else "Not available"

    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task 4 · Sentiment and emotion analysis</title>
<style>
:root{{--ink:#172923;--muted:#62736b;--line:#dce6e0;--green:#23836d;--purple:#8061a8;--red:#d06a5f;--paper:#f0f4f1;--white:#fff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1050px;margin:28px auto;padding:0 20px 48px}}header,.card{{background:var(--white);border:1px solid var(--line);border-radius:16px;padding:24px 28px;margin:0 0 16px;box-shadow:0 5px 20px #17372a0a}}
.eyebrow{{color:var(--green);font-size:12px;font-weight:750;letter-spacing:.12em;text-transform:uppercase}}h1{{font-size:clamp(28px,5vw,42px);line-height:1.12;margin:7px 0}}h2{{font-size:22px;margin:0 0 14px}}h3{{font-size:17px;margin:18px 0 8px}}p{{margin:8px 0}}.muted,.note{{color:var(--muted);font-size:14px}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}}.metric{{background:#f4f8f5;border-radius:12px;padding:14px 16px}}.metric strong{{display:block;font-size:26px}}.metric span{{color:var(--muted);font-size:13px}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.chart{{width:100%;height:auto;margin:8px 0}}.label{{fill:#30433b;font:13px system-ui,sans-serif}}.value{{fill:#52635b;font:12px system-ui,sans-serif}}table{{border-collapse:collapse;width:100%;margin:8px 0}}th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}th{{background:#f6f9f7}}.callout{{background:#f2f7f4;border-left:4px solid var(--green);border-radius:8px;padding:13px 16px}}footer{{color:var(--muted);font-size:13px;padding:8px 4px}}@media(max-width:720px){{.grid{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr}}header,.card{{padding:19px}}}}
</style></head><body><main>
<header><div class="eyebrow">Task 4 · NLP sentiment analysis</div><h1>What reviewers are feeling</h1>
<p>Lexicon-based analysis of {len(analyzed):,} text entries from <strong>{html.escape(source_name)}</strong>.</p>
<p class="muted">This report contains aggregate counts only; individual review text is not reproduced.</p></header>
<section class="card"><h2>Sentiment overview</h2><div class="grid">
<div class="metric"><strong>{len(analyzed):,}</strong><span>reviews analyzed</span></div>
<div class="metric"><strong>{sentiment_counts['positive']:,}</strong><span>positive predictions</span></div>
<div class="metric"><strong>{sentiment_counts['neutral']:,}</strong><span>neutral predictions</span></div>
<div class="metric"><strong>{sentiment_counts['negative']:,}</strong><span>negative predictions</span></div></div>
<div class="two"><div><h3>Predicted sentiment</h3>{svg_bars('Predicted sentiment counts', sentiment_order, '#23836d')}</div>
<div><h3>Emotion signals</h3>{svg_bars('Emotion word matches by review', emotion_order, '#8061a8')}</div></div>
<p class="note">Average sentiment score: {average_score:+.3f} on a scale from −1 (negative) to +1 (positive). Emotion bars count reviews containing at least one lexicon word for each emotion; a review can match multiple emotions.</p></section>
<section class="card"><h2>Most frequent terms</h2><div class="two"><div><h3>Positive predictions</h3>{word_table(top_positive)}</div><div><h3>Negative predictions</h3>{word_table(top_negative)}</div></div></section>
<section class="card"><h2>Validation and interpretation</h2><div class="grid">
<div class="metric"><strong>{len(labelled):,}</strong><span>rows with source labels</span></div>
<div class="metric"><strong>{accuracy_text}</strong><span>accuracy across labelled rows</span></div>
<div class="metric"><strong>{decisive_accuracy_text}</strong><span>accuracy excluding neutral predictions</span></div>
<div class="metric"><strong>{len(decisive):,}/{len(labelled):,}</strong><span>labelled rows with decisive predictions</span></div></div>
<div class="callout"><p><strong>Method:</strong> labeled datasets use a negation-aware Naive Bayes bag-of-words classifier with stratified 5-fold out-of-fold validation; low-confidence predictions are called neutral. Unlabelled datasets use the transparent polarity lexicon and neutral threshold. Emotion signals use word lists for joy, trust, anger, sadness, fear, disgust, anticipation, and surprise.</p><p>{evaluation_note}</p>
<p>Lexicon matches can miss sarcasm, context, domain-specific meanings, and mixed opinions. Use these counts as a screening summary, not as a substitute for human review or a full contextual language model.</p></div></section>
<footer>Generated by sentiment_analysis.py. Review source: {html.escape(source_name)}. Dataset attribution: Kotzias (2015), <a href="https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences">UCI Sentiment Labelled Sentences</a>, CC BY 4.0. No names, account details, or raw review text are included in this report.</footer>
</main></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return {"reviews": len(analyzed), "positive": sentiment_counts["positive"], "neutral": sentiment_counts["neutral"],
            "negative": sentiment_counts["negative"], "accuracy": accuracy, "report": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/reviews.csv"), help="CSV/TSV containing review text")
    parser.add_argument("--output", type=Path, default=Path("sentiment_report.html"), help="HTML report path")
    parser.add_argument("--source-name", default="review dataset", help="Friendly source label for the report")
    parser.add_argument("--text-column", help="Text column name; inferred from common names when omitted")
    parser.add_argument("--label-column", help="Optional source label/rating column; inferred when omitted")
    args = parser.parse_args()
    result = build_report(args.input, args.output, args.source_name, args.text_column, args.label_column)
    print(f"Sentiment report created: {result['report']}")
    print(f"Reviews: {result['reviews']} | positive: {result['positive']} | neutral: {result['neutral']} | negative: {result['negative']}")
    if result["accuracy"] is not None:
        print(f"5-fold out-of-fold accuracy against supplied labels: {result['accuracy']:.1%}")


if __name__ == "__main__":
    main()
