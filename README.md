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

## License

MIT. See [`LICENSE`](LICENSE).
