# Task 4: Sentiment and Emotion Analysis

This project classifies review text as positive, neutral, or negative and counts
word-level signals for eight emotions. It generates a static HTML report with
sentiment and emotion charts, frequent terms, optional label accuracy, and
method limitations. Individual review text is not copied into the report.

## Run it

The script uses only the Python standard library. From this folder, run:

```powershell
py sentiment_analysis.py --input data/reviews.csv --output sentiment_report.html --source-name "Review dataset"
```

The CSV needs a header row and a text column named `review`, `text`, `sentence`,
`comment`, or `content`. If it also has a `label`, `sentiment`, or `rating`
column, the report compares predictions with those labels. You can specify
different columns with `--text-column` and `--label-column`.

Open `sentiment_report.html` in a browser after the command completes.

## Method

- A transparent word lexicon assigns positive and negative polarity.
- A short negation window handles phrases such as “not good.” Intensifiers and
  diminishers adjust word scores.
- Scores near zero are classed as neutral.
- A separate word lexicon counts joy, trust, anger, sadness, fear, disgust,
  anticipation, and surprise signals.

These are lightweight, interpretable rules. They can miss sarcasm, context,
domain-specific wording, and mixed opinions. Treat the results as a screening
summary rather than a definitive measure of public opinion.

## Data source and privacy

This demo uses the Amazon cellphone-review subset from the UCI Sentiment
Labelled Sentences dataset: 1,000 sentences, with 500 positive and 500 negative
labels. The source dataset does not contain neutral ground-truth labels, so
neutral predictions cannot be evaluated against this sample.

Attribution: Kotzias, D. (2015). *Sentiment Labelled Sentences* [Dataset]. UCI
Machine Learning Repository. https://doi.org/10.24432/C57604. Licensed under
CC BY 4.0. The included CSV is a format conversion with a source column added.

The report is aggregate-only. Do not add private review exports or the
separate business-leads workbook to this repository.
