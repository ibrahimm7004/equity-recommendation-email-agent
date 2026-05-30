# Equity Email AI Tool

Python CLI tool that generates professional equity recommendation emails by combining:

1. **Live market data** — price move or earnings results via yfinance
2. **Factsheet analysis** for the current holding (rating, Morningstar stars, risks, valuation)
3. **Factsheet analysis** for the replacement stock (rating, stars, strengths, upside)
4. **OpenAI** to synthesise everything into a client-ready email

## 1) Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 2) Configure

Open `config.py` and set:
- `OPENAI_API_KEY`
- `FACTSHEETS_DIR` — folder containing your equity factsheet PDFs
- (optional) `OPENAI_MODEL`, `OUTPUT_DIR`, match settings

## 3) Run

```powershell
python equity_email_tool.py
```

You will be prompted for:

| Prompt | Example | Required |
|---|---|---|
| Company name | `ABB` | Yes |
| ISIN | `CH0012221716` | Yes |
| Ticker for market data | `ABBN.SW` | Optional (skips live data if blank) |
| Reason | `Move` or `Earnings` | Yes |
| Email language | `English` | Yes |
| Recommendation | `Hold` or `Sell & Buy` | Yes |
| Target company name | `Alphabet` | Only for Sell & Buy |
| Target ISIN | `US02079K3059` | Only for Sell & Buy |
| Target ticker | `GOOGL` | Optional |

## How it works

### Reason: Move
Fetches the previous-day price change and recent news headlines via yfinance.
OpenAI writes one sentence explaining the move (e.g. *"ABB increased by +7.0%
yesterday, driven by stronger-than-expected industrial orders."*).

### Reason: Earnings
Fetches the latest earnings results (EPS actual vs estimate, revenue) and the
market reaction via yfinance. OpenAI writes one sentence summarising the earnings
and price reaction.

### Sell & Buy flow
1. Fetches live data (move or earnings) for the primary stock.
2. Reads the **primary stock factsheet** and extracts sell signals (rating, Morningstar stars, fair value, risks).
3. Reads the **replacement stock factsheet** and extracts buy signals (rating, stars, upside, strengths, portfolio fit).
4. Builds the email with all three sections.

### Hold flow
Same live-data step, then uses the factsheet to explain why the stock should be held.

## Terminal logging

The tool prints detailed logs so you can verify what data was used:

```
[INFO] Company: ABB
[INFO] ISIN: CH0012221716
[INFO] Reason: move
[INFO] Action: sell_buy
[INFO] Market data source: yfinance
[INFO] Last close date: 2026-05-27
[INFO] Previous close date: 2026-05-26
[INFO] Headlines used:
  - ABB reports strong Q1 results
  - Industrial stocks rally on orders data
[INFO] Move explanation generated.
[INFO] Matched primary stock to: ABB Equity Factsheet.pdf
[INFO] Sell analysis complete (rating: Sell, stars: 2)
[INFO] Buy analysis complete (rating: Buy, stars: 4)
```

## Notes

- If yfinance is unavailable or no ticker is entered, the email is still generated without the price-move/earnings paragraph.
- The ISIN you enter is used in the email body (replacing the placeholder).
- Factsheet PDFs are matched by fuzzy name matching against the company name you enter.
- Generated emails are saved as timestamped `.txt` files in `output_emails/`.
