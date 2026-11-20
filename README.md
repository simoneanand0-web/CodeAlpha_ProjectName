# Retail Sales Dashboard

A responsive portfolio dashboard that turns sample monthly sales data into a compact view of revenue, order volume, average order value, monthly movement, and category mix.

## Preview

Open [`index.html`](index.html) in a browser. No build step, server, package install, or external library is required. The reporting-period selector updates the summary metrics, category breakdown, and monthly chart.

## Project files

- `index.html` — dashboard layout, styling, calculations, and SVG charts
- `data/sales-2025.csv` — illustrative monthly revenue and order counts

## Data note

This repository contains synthetic example data for demonstration. It does not represent a real company or customer records. Category shares (Electronics 38%, Home 27%, Apparel 21%, Other 14%) are illustrative and are defined in the dashboard source.

## Customize

Replace the monthly records in `data/sales-2025.csv` and the matching `data` array in `index.html`. Update the `mix` values for your category shares; category shares should sum to 1. The current dashboard is dependency-free and can be hosted as a static page, including with GitHub Pages.

## Task 2: Exploratory Data Analysis

Open the [EDA report](task2-eda/EDA_Report.html) or view the [Python analysis](task2-eda/eda_analysis.py). The report includes dataset questions, structure, patterns, a hypothesis test, and data-quality recommendations.

The report contains aggregate results only. The source leads workbook and individual lead records are not included in this repository. To regenerate it locally, install `pandas`, `numpy`, and `openpyxl`, then run:

```powershell
python task2-eda/eda_analysis.py "path/to/your/workbook.xlsx" --output task2-eda/EDA_Report.html
```

## Task 4: Sentiment and Emotion Analysis

Open the [sentiment report](task4-sentiment/index.html) or view the [analysis script](task4-sentiment/sentiment_analysis.py). This project classifies review text as positive, neutral, or negative and summarizes eight lexicon-based emotion signals. It uses an attributed public Amazon cellphone-review sample; the user's business-leads workbook is not included.

To regenerate the report, run:

```powershell
py task4-sentiment/sentiment_analysis.py --input task4-sentiment/data/amazon_cells_labelled.csv --output task4-sentiment/index.html --source-name "UCI Amazon cellphone reviews"
```

The UCI source has positive and negative labels only, so the report notes that neutral predictions have no neutral source labels for validation. See the Task 4 README for the dataset citation and license.

## License

MIT. See [`LICENSE`](LICENSE).
