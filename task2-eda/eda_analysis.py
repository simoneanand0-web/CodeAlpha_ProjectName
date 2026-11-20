#!/usr/bin/env python3
"""Create an aggregate-only EDA report for a supplied Excel lead export.

The workbook is read locally. The HTML output contains aggregate statistics and
charts only; it does not include individual names, phones, addresses, or URLs.
Requirements: pandas, numpy, openpyxl.
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd


def text_values(series: pd.Series) -> pd.Series:
    """Return trimmed text values, treating empty strings as missing."""
    values = series.astype("string").str.strip()
    return values.mask(values.eq(""), pd.NA)


def duplicate_extra_count(series: pd.Series) -> int:
    values = text_values(series).dropna().str.casefold()
    return int(values.duplicated().sum())


def spearman_permutation(rating: pd.Series, reviews: pd.Series, permutations: int = 20_000):
    """Spearman rho with a two-sided permutation p-value that preserves ties."""
    pairs = pd.DataFrame({"rating": rating, "reviews": reviews}).apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if len(pairs) < 3:
        return {"n": len(pairs), "rho": math.nan, "p": math.nan, "permutations": 0}

    x = pairs["rating"].rank(method="average").to_numpy(dtype=float)
    y = pairs["reviews"].rank(method="average").to_numpy(dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    if denominator == 0:
        return {"n": len(pairs), "rho": math.nan, "p": math.nan, "permutations": 0}

    rho = float(np.dot(xc, yc) / denominator)
    rng = np.random.default_rng(20261120)
    extreme = 0
    for _ in range(permutations):
        permuted_y = rng.permutation(yc)
        permuted_rho = abs(float(np.dot(xc, permuted_y) / denominator))
        extreme += permuted_rho >= abs(rho) - 1e-12
    p_value = (extreme + 1) / (permutations + 1)
    return {"n": len(pairs), "rho": rho, "p": p_value, "permutations": permutations}


def svg_bar_chart(title: str, items: list[tuple[str, int]], denominator: int) -> str:
    width = 820
    row_height = 34
    top = 42
    height = top + row_height * len(items) + 20
    label_x = 190
    bar_x = 205
    bar_width = 475
    max_value = max((value for _, value in items), default=1) or 1
    pieces = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<title>{html.escape(title)}</title>',
        '<style>.grid{stroke:#e6ece9;stroke-width:1}.axis{fill:#64716c;font:12px sans-serif}.label{fill:#24322c;font:13px sans-serif}.bar{fill:#23836d}.value{fill:#32433b;font:12px sans-serif}</style>',
    ]
    for i, (label, value) in enumerate(items):
        y = top + i * row_height
        width_value = max(0, (value / max_value) * bar_width)
        pct = value / denominator * 100 if denominator else 0
        pieces.append(f'<text class="label" x="{label_x}" y="{y + 17}" text-anchor="end">{html.escape(label)}</text>')
        pieces.append(f'<rect class="bar" x="{bar_x}" y="{y + 4}" width="{width_value:.1f}" height="18" rx="4"/>')
        pieces.append(f'<text class="value" x="{bar_x + bar_width + 10}" y="{y + 18}">{value} ({pct:.1f}%)</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def svg_scatter_chart(pairs: pd.DataFrame) -> str:
    width, height = 820, 380
    left, right, top, bottom = 75, 28, 24, 55
    plot_w, plot_h = width - left - right, height - top - bottom
    max_reviews = max(float(pairs["Reviews"].max()), 10.0)
    max_log = max(3.0, math.ceil(math.log10(max_reviews)))
    y_min, y_max = 3.0, 5.05

    def x_pos(value: float) -> float:
        return left + (math.log10(max(float(value), 1.0)) / max_log) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - float(value)) / (y_max - y_min) * plot_h

    pieces = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Business rating versus review count on a logarithmic review axis">',
        '<title>Rating versus review count</title>',
        '<style>.grid{stroke:#e6ece9;stroke-width:1}.axis{fill:#64716c;font:12px sans-serif}.dot{fill:#23836d;fill-opacity:.58;stroke:#fff;stroke-width:.6}</style>',
    ]
    for tick in [3.0, 3.5, 4.0, 4.5, 5.0]:
        y = y_pos(tick)
        pieces.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
        pieces.append(f'<text class="axis" x="{left-12}" y="{y+4:.1f}" text-anchor="end">{tick:.1f}</text>')
    for power in range(max_log + 1):
        value = 10 ** power
        if value <= max_reviews * 1.05:
            x = x_pos(value)
            pieces.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}"/>')
            pieces.append(f'<text class="axis" x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle">{value:,}</text>')
    for rating, reviews in pairs[["Rating", "Reviews"]].dropna().itertuples(index=False, name=None):
        pieces.append(f'<circle class="dot" cx="{x_pos(float(reviews)):.1f}" cy="{y_pos(float(rating)):.1f}" r="5"/>')
    pieces.extend([
        f'<text class="axis" x="{left + plot_w/2:.1f}" y="{height-8}" text-anchor="middle">Review count (log scale)</text>',
        f'<text class="axis" transform="translate(18 {top + plot_h/2:.1f}) rotate(-90)" text-anchor="middle">Rating (out of 5)</text>',
        "</svg>",
    ])
    return "".join(pieces)


def build_report(input_path: Path, output_path: Path) -> dict:
    workbook = pd.ExcelFile(input_path)
    sheet_name = workbook.sheet_names[0]
    raw = pd.read_excel(input_path, sheet_name=sheet_name)
    total_rows, total_columns = raw.shape

    empty_row_mask = ~raw.notna().any(axis=1)
    named_mask = text_values(raw["Name"]).notna() if "Name" in raw else raw.notna().any(axis=1)
    records = raw.loc[named_mask].copy()
    record_count = len(records)

    def valid_count(column: str) -> int:
        if column not in records:
            return 0
        series = records[column]
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype):
            return int(text_values(series).notna().sum())
        return int(series.notna().sum())

    def missing_count(column: str) -> int:
        return record_count - valid_count(column)

    empty_columns = [column for column in raw.columns if raw[column].isna().all()]
    blank_unnamed_columns = sum(str(column).startswith("Unnamed:") for column in empty_columns)
    blank_category = text_values(records["Category"]).isna() if "Category" in records else pd.Series(False, index=records.index)
    category_counts = text_values(records["Category"]).dropna().value_counts().sort_values(ascending=False)
    category_plot_items = [(str(label), int(value)) for label, value in category_counts.items()]
    if int(blank_category.sum()):
        category_plot_items.append(("Missing / blank", int(blank_category.sum())))

    rating = pd.to_numeric(records.get("Rating", pd.Series(dtype=float)), errors="coerce")
    reviews = pd.to_numeric(records.get("Reviews", pd.Series(dtype=float)), errors="coerce")
    rating_values = rating.dropna()
    review_values = reviews.dropna()
    rating_review = spearman_permutation(rating, reviews)
    pairs = pd.DataFrame({"Rating": rating, "Reviews": reviews}).dropna()

    q1 = float(review_values.quantile(0.25)) if len(review_values) else math.nan
    q3 = float(review_values.quantile(0.75)) if len(review_values) else math.nan
    review_upper_fence = q3 + 1.5 * (q3 - q1) if len(review_values) else math.nan
    review_high_outliers = int((review_values > review_upper_fence).sum()) if len(review_values) else 0

    status_counts = text_values(records.get("Status", pd.Series(dtype="string"))).value_counts()
    new_count = int(status_counts.get("New", 0))
    phone1 = pd.to_numeric(records.get("Phone", pd.Series(dtype=float)), errors="coerce")
    phone2 = pd.to_numeric(records.get("Phone.1", pd.Series(dtype=float)), errors="coerce")
    phone_available = int((phone1.notna() | phone2.notna()).sum())

    last_contacted = pd.to_datetime(records.get("Last Contacted", pd.Series(dtype="datetime64[ns]")), errors="coerce").dropna()
    update_values = text_values(records.get("update", pd.Series(dtype="string"))).dropna()

    duplicates = {
        column: duplicate_extra_count(records[column]) if column in records else 0
        for column in ["Name", "Maps URL", "Address"]
    }
    exact_duplicate_rows = int(records.duplicated().sum())
    rating_out_of_range = int((~rating_values.between(0, 5)).sum())
    negative_reviews = int((review_values < 0).sum())
    replacement_character_updates = sum("\ufffd" in str(value) for value in update_values)

    missing_items = [
        ("Last contacted", missing_count("Last Contacted")),
        ("Tags", missing_count("Tags")),
        ("Notes", missing_count("Notes")),
        ("Update note", missing_count("update")),
        ("Phone fields", record_count - phone_available),
        ("Rating", missing_count("Rating")),
        ("Reviews", missing_count("Reviews")),
        ("Category", missing_count("Category")),
        ("Address", missing_count("Address")),
        ("Maps URL", missing_count("Maps URL")),
        ("Status", missing_count("Status")),
    ]

    completeness_chart = svg_bar_chart("Missing values across named lead records", missing_items, record_count)
    category_chart = svg_bar_chart("Business category counts", category_plot_items, record_count)
    scatter_chart = svg_scatter_chart(pairs) if len(pairs) else "<p>No complete rating and review pairs are available.</p>"

    rating_median = float(rating_values.median()) if len(rating_values) else math.nan
    rating_high_count = int((rating_values >= 4.8).sum()) if len(rating_values) else 0
    review_median = float(review_values.median()) if len(review_values) else math.nan
    review_mean = float(review_values.mean()) if len(review_values) else math.nan
    review_max = float(review_values.max()) if len(review_values) else math.nan
    top_category_label = str(category_counts.index[0]) if len(category_counts) else "No category values"
    top_category_count = int(category_counts.iloc[0]) if len(category_counts) else 0
    top_category_share = (100 * top_category_count / int(category_counts.sum())) if int(category_counts.sum()) else 0.0

    def pct(value: int, base: int = record_count) -> str:
        return f"{100 * value / base:.1f}%" if base else "n/a"

    category_rows = "".join(
        f"<tr><td>{html.escape(str(label))}</td><td>{int(value)}</td><td>{pct(int(value))}</td></tr>"
        for label, value in category_counts.items()
    )
    data_dictionary = [
        ("Name", "Text identifier", f"{valid_count('Name')} / {record_count} populated"),
        ("Phone (two source columns named Phone)", "Numeric in Excel import", f"{phone_available} / {record_count} records have at least one number"),
        ("Status", "Text category", f"{new_count} marked New; {missing_count('Status')} missing"),
        ("Tags and Notes", "Text fields", "Both columns are entirely empty"),
        ("Last Contacted", "Date", f"{len(last_contacted)} populated; {last_contacted.nunique()} distinct date(s)"),
        ("Rating", "Numeric, 0–5 expected", f"{len(rating_values)} values; range {rating_values.min():.1f}–{rating_values.max():.1f}" if len(rating_values) else "No values"),
        ("Reviews", "Numeric count", f"{len(review_values)} values; range {review_values.min():.0f}–{review_values.max():.0f}" if len(review_values) else "No values"),
        ("Category", "Text category", f"{len(category_counts)} nonblank labels; {missing_count('Category')} blank"),
        ("Address and Maps URL", "Text / URL", f"{valid_count('Address')} addresses; {valid_count('Maps URL')} map links"),
        ("Update", "Free text", f"{len(update_values)} populated notes"),
        ("Other columns", "Unused", f"{blank_unnamed_columns} empty unnamed columns; {len(empty_columns) - blank_unnamed_columns} other fully empty columns"),
    ]
    dictionary_rows = "".join(
        f"<tr><th scope='row'>{html.escape(label)}</th><td>{html.escape(kind)}</td><td>{html.escape(detail)}</td></tr>"
        for label, kind, detail in data_dictionary
    )

    if math.isnan(rating_review["rho"]):
        test_text = "The rating and review-count test could not be run because there were not enough complete pairs."
    else:
        test_text = (
            f"On {rating_review['n']} complete records, Spearman’s rank correlation was "
            f"ρ = {rating_review['rho']:.3f}. A two-sided permutation test with "
            f"{rating_review['permutations']:,} shuffles gave p = {rating_review['p']:.4f}. "
            "This is evidence of a modest negative association in this sheet. It does not show that review count causes rating changes, and the rating values are rounded with many ties."
        )

    html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task 2 EDA · Local business leads</title>
<style>
:root{{--ink:#192923;--muted:#5c6d65;--line:#dce5e0;--green:#176b55;--soft:#f2f7f4;--white:#fff;--amber:#fff7e8}}
*{{box-sizing:border-box}}body{{margin:0;background:#edf2ef;color:var(--ink);font:16px/1.58 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1040px;margin:32px auto;padding:0 20px 48px}}header,.card{{background:var(--white);border:1px solid var(--line);border-radius:16px;padding:24px 28px;margin-bottom:16px;box-shadow:0 5px 20px #17372a0a}}
h1{{font-size:clamp(28px,5vw,42px);line-height:1.12;margin:6px 0 10px;letter-spacing:-.03em}}h2{{font-size:22px;margin:0 0 12px}}h3{{font-size:17px;margin:18px 0 8px}}p{{margin:8px 0;color:#34443d}}.eyebrow{{color:var(--green);font-weight:700;text-transform:uppercase;letter-spacing:.12em;font-size:12px}}
.subtle,.note{{color:var(--muted);font-size:14px}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0}}.metric{{background:var(--soft);border-radius:12px;padding:14px 16px}}.metric strong{{display:block;font-size:27px;line-height:1.2}}.metric span{{display:block;color:var(--muted);font-size:13px;margin-top:5px}}
ol,ul{{padding-left:22px}}li{{margin:5px 0}}table{{width:100%;border-collapse:collapse;margin:10px 0}}th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}}th{{font-weight:650;background:#f7faf8}}.chart{{display:block;width:100%;height:auto;margin:10px 0}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.callout{{border-left:4px solid var(--green);background:var(--soft);padding:14px 16px;border-radius:8px}}.warning{{border-left-color:#bd8427;background:var(--amber)}}code{{background:#f0f4f2;border-radius:5px;padding:2px 5px}}footer{{padding:8px 4px;color:var(--muted);font-size:13px}}
@media(max-width:740px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.two{{grid-template-columns:1fr}}header,.card{{padding:19px}}main{{margin:14px auto}}}}
</style></head><body><main>
<header><div class="eyebrow">Task 2 · Exploratory data analysis</div><h1>Local business leads</h1><p>Questions, data structure, patterns, a statistical hypothesis test, and data-quality findings from the supplied workbook.</p><p class="subtle">Aggregate report only. It contains no individual names, phone numbers, addresses, or map links. The source workbook is not copied into this folder.</p></header>

<section class="card"><h2>Questions to ask first</h2><ol>
<li>How many usable lead records does the sheet actually contain, and does that match its “106 leads” label?</li>
<li>Which business categories are represented well enough to compare?</li>
<li>Are ratings and review counts complete and informative, or skewed and clustered?</li>
<li>Does the sheet track contact activity consistently?</li>
<li>Can it tell us which leads responded, qualified, or converted?</li></ol></section>

<section class="card"><h2>Dataset snapshot</h2><div class="grid">
<div class="metric"><strong>{total_rows:,}</strong><span>worksheet rows</span></div>
<div class="metric"><strong>{record_count:,}</strong><span>records with a name</span></div>
<div class="metric"><strong>{total_columns:,}</strong><span>columns in the worksheet</span></div>
<div class="metric"><strong>{len(pairs):,}</strong><span>complete rating / review pairs</span></div>
</div><p>The sheet has {int(empty_row_mask.sum())} fully blank row(s). It contains {len(empty_columns)} entirely empty columns, including {blank_unnamed_columns} unnamed columns. The “106 leads” label matches the {new_count} rows marked “New,” but the sheet has {record_count} records with names; {missing_count('Status')} named records have no status.</p>
<p class="subtle">The analysis uses the {record_count} named records as its base. Missing-field percentages use that base. The rating/review test uses complete pairs only.</p></section>

<section class="card"><h2>Variables and data types</h2><table><thead><tr><th>Field group</th><th>Type / role</th><th>Coverage in named records</th></tr></thead><tbody>{dictionary_rows}</tbody></table>
<p>The two source columns both have the header <code>Phone</code>; the Python reader labels the second one <code>Phone.1</code>. Both were imported as numbers. Confirm what each represents and store phone identifiers as text before cleaning them, so leading zeros or formatting are not lost.</p></section>

<section class="card"><h2>Patterns and trends</h2><div class="two"><div><h3>Business category</h3>{category_chart}<p class="note">{html.escape(top_category_label)} accounts for {top_category_count} of {int(category_counts.sum())} labeled categories ({top_category_share:.1f}%). The remaining categories have small counts, so category-level comparisons would be unstable.</p></div><div><h3>Missing fields</h3>{completeness_chart}<p class="note">The free-text Update, Tags, and Notes fields are especially incomplete. Phone availability means at least one of the two phone columns is populated.</p></div></div>
<ul><li><strong>Ratings:</strong> {len(rating_values)} records have a rating from {rating_values.min():.1f} to {rating_values.max():.1f}; the median is {rating_median:.1f}. {rating_high_count} of {len(rating_values)} ratings ({100*rating_high_count/len(rating_values):.1f}%) are at least 4.8, so the distribution has a ceiling effect.</li>
<li><strong>Review counts:</strong> the median is {review_median:.0f}, compared with a mean of {review_mean:.1f}, and the maximum is {review_max:.0f}. This right-skew means the mean is pulled upward by a small number of high-review listings.</li>
<li><strong>Potential high outliers:</strong> {review_high_outliers} review counts exceed the usual 1.5×IQR upper fence ({review_upper_fence:.1f}). These are candidates to inspect, not automatic errors.</li>
<li><strong>Contact activity:</strong> {phone_available} of {record_count} named records have a phone value. Only {len(last_contacted)} records have a Last Contacted date, and those dates all share one calendar day. All {new_count} populated statuses are “New.”</li>
<li><strong>Duplicates:</strong> no duplicate nonblank names, map URLs, or addresses were found after trimming whitespace and ignoring case. Exact duplicate named rows: {exact_duplicate_rows}.</li></ul></section>

<section class="card"><h2>Hypothesis test: ratings and review counts</h2><div class="callout"><p><strong>Question:</strong> Are higher review counts associated with higher public ratings?</p><p><strong>H₀:</strong> rating and review count have no monotonic association. <strong>H₁:</strong> they have a monotonic association.</p><p>{test_text}</p></div>{scatter_chart}<p class="note">Method: Spearman rank correlation, two-sided permutation test ({rating_review['permutations']:,} shuffles, fixed seed). Average ranks handle tied ratings; complete records only. This is an association within this workbook, not a causal effect or a measure of lead conversion.</p></section>

<section class="card"><h2>Data issues to address</h2><ol>
<li>Remove the fully blank row and the {len(empty_columns)} unused columns. Reconcile the file’s 106-lead label with its {record_count} named records.</li>
<li>Give the duplicate Phone headers distinct names and confirm whether they mean primary and secondary contact numbers. Keep phone values as text.</li>
<li>Standardize Category and Status values. Use a controlled set of update outcomes instead of free-text notes, and reconcile notes with status.</li>
<li>Confirm what Last Contacted means. Its {len(last_contacted)} populated values all share one date, which may represent an export/default date rather than separate contact events.</li>
<li>Add explicit outcome fields such as contact attempt, response, qualified, and converted if the goal is to evaluate the lead funnel.</li>
<li>Keep missing ratings and review counts as missing until a source can validate them. Do not replace them with zero.</li></ol>
<div class="callout warning"><p><strong>What this dataset cannot answer:</strong> every populated Status is “New,” and there are no response, qualification, or conversion outcomes. It cannot support conversion-rate analysis or identify which categories produce successful leads.</p></div></section>

<section class="card"><h2>Assumptions and limitations</h2><ul>
<li>The rows appear to represent local business listings, based on the available fields. Confirm this interpretation with the source owner.</li>
<li>The workbook’s collection method and selection criteria are not documented. Findings describe this sheet and should not be generalized to all local businesses.</li>
<li>Category labels are left as supplied after trimming surrounding whitespace. No category mapping or imputation was invented.</li>
<li>Ratings are rounded and concentrated near five stars. Review counts are strongly skewed. Both points limit how much the statistical result should be used for decisions.</li>
<li>{replacement_character_updates} free-text update value(s) contain a text replacement character and should be checked against the original source.</li>
</ul></section>
<footer>Generated from the first worksheet of the supplied Excel file. Only aggregate summaries and non-identifying charts are included.</footer>
</main></body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_report, encoding="utf-8")
    return {
        "sheet": sheet_name,
        "rows": total_rows,
        "records": record_count,
        "columns": total_columns,
        "report": str(output_path),
        "spearman_rho": rating_review["rho"],
        "permutation_p": rating_review["p"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the supplied Excel workbook")
    parser.add_argument("--output", type=Path, default=Path("eda_report.html"), help="HTML report path")
    args = parser.parse_args()
    result = build_report(args.input, args.output)
    print(f"EDA report created: {result['report']}")
    print(f"Named records: {result['records']} | worksheet rows: {result['rows']} | columns: {result['columns']}")
    if not math.isnan(result["spearman_rho"]):
        print(f"Rating/review Spearman rho: {result['spearman_rho']:.3f}; permutation p: {result['permutation_p']:.4f}")


if __name__ == "__main__":
    main()
