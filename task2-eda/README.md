# Task 2: Exploratory Data Analysis

This folder contains an aggregate-only EDA report and the Python script that generated it. The source workbook is not included. The report does not include individual business names, phone numbers, addresses, or map links.

## Recreate the report

Install Python 3 with `pandas`, `numpy`, and `openpyxl`. Then run:

```powershell
python eda_analysis.py "C:\path\to\leadgrab-106-leads.csv.xlsx" --output EDA_Report.html
```

Open `EDA_Report.html` in a browser. The script reads the first worksheet and writes aggregate summaries and SVG charts only.

## Key result

The worksheet has 109 rows, including one fully blank row, and 108 named records. There are 106 records marked `New` and two named records without a status. For the 99 complete rating/review pairs, the Spearman rank correlation is approximately -0.305; a two-sided permutation test gives p ≈ 0.0024. This is an association in this sheet, not evidence of causation or lead conversion.
