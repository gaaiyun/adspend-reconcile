# adspend-reconcile

Align the spend and revenue numbers your ad platforms report against the revenue your store actually booked, day by day and channel by channel, so you can see exactly where they disagree.

Meta says it drove $19k. Google says $26k. TikTok says $3.4k. Add them up and you "made" $48k from $13k of spend. Your Shopify dashboard shows $45k. So which number is real, and why is each platform's claim off? `adspend-reconcile` answers that with a local, auditable join: no OAuth, no API keys, no data leaving your machine. You export CSVs, it reconciles them.

This is the boring data layer that sits *below* a marketing mix model: the part everyone needs and nobody open-sources. It does not try to be MMM, it tries to make your numbers add up.

## What it does

- **ingest** an ad-platform export (Meta / Google / TikTok) or a store export (Shopify / Stripe) and normalise it into one tidy `date x channel` schema. Column names are matched by platform preset, then by fuzzy similarity, then by an explicit override you supply.
- **reconcile** ad spend against real store revenue on a `date x channel` grid. Every cell is classified:
  - `matched` — spend and store revenue both present, and the platform's self-reported revenue agrees within a tolerance.
  - `discrepant` — both present, but the platform's claimed revenue diverges beyond the tolerance. This is the "why don't the numbers line up" bucket.
  - `unattributed` — store revenue with no matching spend (organic / direct), or spend with no store revenue traced to it.
- Apply a **profit model** (COGS, shipping, processing fees, refunds) to turn revenue into contribution margin, then report **contribution-margin ROAS**, gross ROAS, per-channel marginal profit, and a single **blended MER** for the whole account.
- **report** the same run with the largest per-cell discrepancies ranked, so you know which days and channels to investigate first.

Everything runs offline against CSV files. The tool ships with a synthetic sample dataset so you can try it before touching real data.

## Install

```
git clone https://github.com/gaaiyun/adspend-reconcile.git
cd adspend-reconcile
pip install -e .
```

Requires Python 3.10+. Dependencies: `pandas`, `duckdb`, `typer`, `rich`.

## 30-second quickstart

```
# 1. write the bundled synthetic sample data (no real account needed)
adspend-reconcile sample sample_data

# 2. reconcile three ad exports against the store export, with a profit model
adspend-reconcile reconcile \
    --store sample_data/shopify_revenue_sample.csv \
    --ads meta:sample_data/meta_ads_sample.csv \
    --ads google:sample_data/google_ads_sample.csv \
    --ads tiktok:sample_data/tiktok_ads_sample.csv \
    --cogs 0.35 --shipping 0.08 --fee-rate 0.029 --fee-flat 0.30
```

If you have not installed the entry point, every command also works as
`python -m adspend_reconcile.cli ...`. On Windows, prefix with
`PYTHONIOENCODING=utf-8` if your console mangles the table borders.

## What the output looks like

The quickstart command above prints (numbers are from the shipped sample data):

```
                             Account summary
┌───────────────────────────┬────────────────────────────────────────────┐
│ metric                    │                                      value │
├───────────────────────────┼────────────────────────────────────────────┤
│ date range                │ 2024-01-01 -> 2024-01-14                   │
│ channels                  │              google, meta, organic, tiktok │
│ cells (date x channel)    │                                         56 │
│   matched                 │                                         12 │
│   discrepant              │                                         30 │
│   unattributed            │                                         14 │
│ total ad spend            │                                  12,801.43 │
│ platform-reported revenue │                                  48,421.45 │
│ store revenue (actual)    │                                  45,344.69 │
│ platform vs store gap     │                                   3,076.76 │
│ contribution margin       │                                  22,953.53 │
│ blended MER               │                                      3.542 │
│ blended CM ROAS           │                                      1.793 │
│ account marginal profit   │                                  10,152.10 │
└───────────────────────────┴────────────────────────────────────────────┘

                                  By channel
┌─────────┬──────────┬───────────────┬───────────────┬───────────────┬────────────┬─────────┬────────────────┐
│ channel │    spend │ platform_rev. │ store_revenue │ contribution. │ gross_roas │ cm_roas │ marginal_prof. │
├─────────┼──────────┼───────────────┼───────────────┼───────────────┼────────────┼─────────┼────────────────┤
│ google  │ 5,736.00 │     25,964.09 │     25,465.42 │     12,821.57 │       4.44 │    2.24 │       7,085.57 │
│ meta    │ 4,476.61 │     19,054.59 │     14,508.13 │      7,398.94 │       3.24 │    1.65 │       2,922.33 │
│ tiktok  │ 2,588.82 │      3,402.77 │      1,813.03 │        913.96 │       0.70 │    0.35 │      -1,674.86 │
│ organic │     0.00 │          0.00 │      3,558.11 │      1,819.06 │        nan │     nan │       1,819.06 │
└─────────┴──────────┴───────────────┴───────────────┴───────────────┴────────────┴─────────┴────────────────┘
```

How to read it:

- **Google** claims $25,964 and the store booked $25,465 — within tolerance, so its days land in `matched`. Contribution-margin ROAS of 2.24 means it clears its variable costs comfortably.
- **Meta** claims $19,054 but the store only booked $14,508 — a 31% over-report, so its days are `discrepant`. It is still profitable after costs (CM ROAS 1.65).
- **TikTok** claims $3,402 against $1,813 of real revenue, and a CM ROAS of 0.35: every dollar spent there currently loses money (marginal profit -$1,675). This is the channel to cut or fix first.
- **organic** has $3,558 of revenue and zero spend, so it is `unattributed` — free traffic that no ad budget paid for. ROAS is `nan` by design (revenue with no spend has no return *on ad spend*), never a misleading infinity.
- **blended MER** of 3.54 is the honest top-line: total store revenue over total spend, with no double-counting of overlapping platform attribution.

`adspend-reconcile report ...` prints the same tables plus a ranked list of the largest `discrepant` cells so you can see which specific days drove the gap. `--out cells.csv` on either command writes the full per-cell table for your own pivots.

## Bringing your own data

Export a daily breakdown from each ad platform and from your store, then point the CLI at them.

- Ad exports are single-platform, so the channel label comes from the preset (`--ads meta:file.csv`) or `--channel`. The tool reads `date`, `spend`, platform-reported `conversions`, and platform-reported `revenue`; it keeps the campaign name as an audit column.
- Store exports need a `date`, a `channel` column (the marketing channel your store attributes the order to), `revenue`, and ideally `orders` and `refunds`.

Presets ship for `meta`, `google`, `tiktok` (ads) and `shopify`, `stripe` (store). If a column is not matched automatically, check the mapping table that `ingest` prints and override it:

```
adspend-reconcile ingest your_export.csv --kind ads --preset meta \
    --map spend="Amount spent (USD)" --map platform_revenue="Purchases conversion value"
```

## How the reconciliation works

The join is a DuckDB full outer join on `(date, channel)`, so both unmatched ad spend and unmatched store revenue survive instead of being silently dropped. Classification compares the platform's self-reported revenue to the store's booked revenue with a relative tolerance (`--tolerance`, default 15%). The profit model is applied to the *store* revenue, because contribution margin should be measured against money that actually arrived:

```
contribution_margin = store_revenue - COGS - shipping - processing_fees - refunds
contribution_margin_roas = contribution_margin / ad_spend
blended_mer = total_store_revenue / total_ad_spend
marginal_profit = contribution_margin - ad_spend
```

Division by zero never produces infinity: a metric with no denominator is reported as `nan` so it cannot quietly distort a ranking or an average.

## Scope and honesty

This tool deliberately does a small thing well. It is a reconciliation and contribution-margin layer, not an attribution engine and not a marketing mix model.

A `attribution` module is included for the four standard *rule-based* splits — first-touch, last-touch, linear, and time-decay — over multi-touch conversion paths. These are useful for showing how much your per-channel credit moves on the *choice of rule alone*, which is one reason platform numbers disagree. They are heuristics, not causal estimates. Statistical attribution (Shapley, Markov), media mix modelling, budget optimisation, and incrementality testing are intentionally out of scope: doing them correctly needs assumptions and validation this tool does not provide, and there are dedicated libraries for each.

## Roadmap

- Weekly / monthly reconciliation grain in addition to daily.
- Currency normalisation across multi-currency exports.
- More store presets (WooCommerce, BigCommerce) and more ad presets (Pinterest, Amazon Ads).
- A `diff` command to compare two reconciliation runs over time.
- Optional anomaly flags for sudden week-over-week gap changes.

These are not built yet. The reconciliation core, the profit model, the CLI, and the sample data are.

## Development

```
pip install -e ".[dev]"
python -m pytest tests/ -q
```

## License

MIT.
