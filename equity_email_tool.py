from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import config

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import yfinance as yf
except ImportError:
    yf = None

from openai import OpenAI


TOKEN_STOPWORDS = {
    "equity",
    "factsheet",
    "fact",
    "sheet",
    "class",
    "inc",
    "corporation",
    "corp",
    "plc",
    "sa",
    "nv",
    "ag",
    "ltd",
    "limited",
    "co",
    "company",
    "common",
    "stock",
    "shares",
    "adr",
    "ordinary",
    "ord",
    "the",
}


@dataclass
class UserInputs:
    stock_name: str
    isin: str
    action: str
    reason: str  # "move" or "earnings"
    language: str
    ticker: str | None = None
    replacement_stock: str | None = None
    replacement_isin: str | None = None
    replacement_ticker: str | None = None


@dataclass
class MarketData:
    ticker: str
    company_name: str
    last_price: float
    previous_close: float
    change_pct: float
    currency: str
    last_close_date: str = ""
    previous_close_date: str = ""
    news_headlines: list[str] = field(default_factory=list)


@dataclass
class EarningsData:
    ticker: str
    company_name: str
    currency: str
    reported_eps: float | None = None
    estimated_eps: float | None = None
    eps_surprise_pct: float | None = None
    revenue: float | None = None
    last_price: float = 0.0
    previous_close: float = 0.0
    change_pct: float = 0.0
    last_close_date: str = ""
    previous_close_date: str = ""
    news_headlines: list[str] = field(default_factory=list)


# ── Text utilities ──────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def significant_tokens(text: str) -> list[str]:
    tokens = normalize_text(text).split()
    return [tok for tok in tokens if tok not in TOKEN_STOPWORDS and not tok.isdigit()]


# ── Factsheet matching ──────────────────────────────────────────────


def file_match_score(stock_query: str, pdf_path: Path) -> float:
    query_norm = normalize_text(stock_query)
    file_norm = normalize_text(pdf_path.stem)

    if not query_norm or not file_norm:
        return 0.0

    score = 0.0

    if query_norm in file_norm:
        score += 45.0

    query_tokens = set(significant_tokens(stock_query))
    file_tokens = set(significant_tokens(pdf_path.stem))
    if query_tokens and file_tokens:
        overlap = len(query_tokens & file_tokens) / len(query_tokens)
        score += overlap * 35.0

    seq = SequenceMatcher(a=query_norm, b=file_norm).ratio()
    score += seq * 20.0

    return score


def find_best_factsheet(stock_query: str, factsheet_dir: Path) -> Path:
    pdf_files = sorted(factsheet_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in factsheet folder: {factsheet_dir}")

    scored = [(file_match_score(stock_query, pdf), pdf) for pdf in pdf_files]
    scored.sort(key=lambda item: item[0], reverse=True)

    best_score, best_pdf = scored[0]
    if len(scored) > 1:
        second_score, second_pdf = scored[1]
        if (best_score - second_score) < 3.0 and best_score < 60.0:
            raise ValueError(
                f'Ambiguous factsheet match for "{stock_query}".\n'
                f"Top candidates are too close:\n"
                f"- {best_pdf.name} (score={best_score:.1f})\n"
                f"- {second_pdf.name} (score={second_score:.1f})\n"
                "Use a more specific stock name."
            )

    if best_score < config.MIN_MATCH_SCORE:
        suggestions = "\n".join(
            f"- {pdf.name} (score={score:.1f})"
            for score, pdf in scored[: max(config.MATCH_SUGGESTIONS, 1)]
        )
        raise ValueError(
            f'Could not confidently match a factsheet for "{stock_query}".\n'
            f"Top candidates:\n{suggestions}"
        )

    return best_pdf


# ── PDF extraction ──────────────────────────────────────────────────


def extract_with_pypdf(pdf_path: Path) -> str:
    if PdfReader is None:
        return ""
    pages_text: list[str] = []
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def extract_with_pdfplumber(pdf_path: Path) -> str:
    if pdfplumber is None:
        return ""
    pages_text: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def extract_with_pymupdf(pdf_path: Path) -> str:
    if fitz is None:
        return ""
    pages_text: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pages_text.append(page.get_text("text") or "")
    return "\n".join(pages_text)


def cleanup_extracted_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> tuple[str, str]:
    extractors: list[tuple[str, Callable[[Path], str]]] = [
        ("pypdf", extract_with_pypdf),
        ("pdfplumber", extract_with_pdfplumber),
        ("pymupdf", extract_with_pymupdf),
    ]

    candidates: list[tuple[int, str, str]] = []
    errors: list[str] = []

    for name, extractor in extractors:
        try:
            raw_text = extractor(pdf_path)
            cleaned = cleanup_extracted_text(raw_text)
            if cleaned:
                candidates.append((len(cleaned), name, cleaned))
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if not candidates:
        detail = "\n".join(errors) if errors else "No extractor produced readable text."
        raise RuntimeError(f"Failed to extract text from {pdf_path.name}.\n{detail}")

    best_len, best_name, best_text = max(candidates, key=lambda item: item[0])
    if best_len < 200:
        raise RuntimeError(
            f"Extracted text from {pdf_path.name} is unexpectedly short ({best_len} chars)."
        )
    return best_text, best_name


# ── Action parsing ──────────────────────────────────────────────────


def parse_action(raw_action: str) -> str:
    action = normalize_text(raw_action)
    if action in {"hold", "h"}:
        return "hold"
    if action in {"sell", "sell buy", "sell and buy", "sell buy a new stock", "s"}:
        return "sell_buy"
    if "sell" in action and "buy" in action:
        return "sell_buy"
    raise ValueError("Action must be Hold or Sell & Buy.")


# ── Market data via yfinance ────────────────────────────────────────


def _extract_headlines(stock: object) -> list[str]:
    """Pull up to 5 news headlines from a yfinance Ticker, tolerating API changes."""
    headlines: list[str] = []
    try:
        raw_news = stock.news or []  # type: ignore[union-attr]
        for article in raw_news[:5]:
            if isinstance(article, dict):
                title = article.get("title", "") or (
                    (article.get("content") or {}).get("title", "")
                )
                if title:
                    headlines.append(title)
    except Exception:
        pass
    return headlines


def fetch_market_data(ticker: str) -> MarketData | None:
    """Fetch previous-day price move and news headlines for *ticker*."""
    if yf is None:
        print("[INFO] yfinance not installed; skipping live market data.")
        return None
    try:
        print(f"[INFO] Market data source: yfinance")
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        hist = stock.history(period="5d")
        last_date = prev_date = ""
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            last_close = float(hist["Close"].iloc[-1])
            last_date = str(hist.index[-1].date())
            prev_date = str(hist.index[-2].date())
        else:
            prev_close = float(info.get("previousClose", 0))
            last_close = float(
                info.get("currentPrice", info.get("regularMarketPrice", 0))
            )

        change_pct = (
            ((last_close - prev_close) / prev_close * 100) if prev_close else 0.0
        )

        print(f"[INFO] Last close date: {last_date or 'N/A'}")
        print(f"[INFO] Previous close date: {prev_date or 'N/A'}")

        headlines = _extract_headlines(stock)
        if headlines:
            print("[INFO] Headlines used:")
            for h in headlines:
                print(f"  - {h}")
        else:
            print("[INFO] No recent news headlines found.")

        return MarketData(
            ticker=ticker,
            company_name=info.get("shortName", ticker),
            last_price=round(last_close, 2),
            previous_close=round(prev_close, 2),
            change_pct=round(change_pct, 2),
            currency=info.get("currency", "USD"),
            last_close_date=last_date,
            previous_close_date=prev_date,
            news_headlines=headlines,
        )
    except Exception as exc:
        print(f"[INFO] Could not fetch market data for {ticker}: {exc}")
        return None


def fetch_earnings_data(ticker: str) -> EarningsData | None:
    """Fetch latest earnings results, price reaction, and news for *ticker*."""
    if yf is None:
        print("[INFO] yfinance not installed; skipping earnings data.")
        return None
    try:
        print(f"[INFO] Earnings data source: yfinance")
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Price data (same as move path — needed for market reaction)
        hist = stock.history(period="5d")
        last_date = prev_date = ""
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            last_close = float(hist["Close"].iloc[-1])
            last_date = str(hist.index[-1].date())
            prev_date = str(hist.index[-2].date())
        else:
            prev_close = float(info.get("previousClose", 0))
            last_close = float(
                info.get("currentPrice", info.get("regularMarketPrice", 0))
            )

        change_pct = (
            ((last_close - prev_close) / prev_close * 100) if prev_close else 0.0
        )

        print(f"[INFO] Last close date: {last_date or 'N/A'}")
        print(f"[INFO] Previous close date: {prev_date or 'N/A'}")

        # Earnings metrics
        reported_eps: float | None = None
        estimated_eps: float | None = None
        eps_surprise_pct: float | None = None
        revenue: float | None = None

        try:
            earnings_hist = stock.earnings_history
            if earnings_hist is not None and len(earnings_hist) > 0:
                latest = earnings_hist.iloc[-1]
                reported_eps = float(latest.get("epsActual", 0)) if latest.get("epsActual") is not None else None
                estimated_eps = float(latest.get("epsEstimate", 0)) if latest.get("epsEstimate") is not None else None
                surprise = latest.get("epsDifference") or latest.get("surprisePercent")
                eps_surprise_pct = float(surprise) if surprise is not None else None
        except Exception:
            pass

        try:
            rev = info.get("totalRevenue") or info.get("revenue")
            revenue = float(rev) if rev is not None else None
        except Exception:
            pass

        print(f"[INFO] Reported EPS: {reported_eps}")
        print(f"[INFO] Estimated EPS: {estimated_eps}")
        print(f"[INFO] EPS surprise: {eps_surprise_pct}")
        print(f"[INFO] Revenue: {revenue}")

        headlines = _extract_headlines(stock)
        if headlines:
            print("[INFO] Headlines used:")
            for h in headlines:
                print(f"  - {h}")
        else:
            print("[INFO] No recent news headlines found.")

        return EarningsData(
            ticker=ticker,
            company_name=info.get("shortName", ticker),
            currency=info.get("currency", "USD"),
            reported_eps=reported_eps,
            estimated_eps=estimated_eps,
            eps_surprise_pct=eps_surprise_pct,
            revenue=revenue,
            last_price=round(last_close, 2),
            previous_close=round(prev_close, 2),
            change_pct=round(change_pct, 2),
            last_close_date=last_date,
            previous_close_date=prev_date,
            news_headlines=headlines,
        )
    except Exception as exc:
        print(f"[INFO] Could not fetch earnings data for {ticker}: {exc}")
        return None


def parse_reason(raw_reason: str) -> str:
    reason = normalize_text(raw_reason)
    if reason in {"move", "m", "price move", "market move"}:
        return "move"
    if reason in {"earnings", "e", "earning", "results", "quarterly"}:
        return "earnings"
    raise ValueError("Reason must be Move or Earnings.")


# ── Prompt builders ─────────────────────────────────────────────────


def build_move_prompt(stock_name: str, language: str, market: MarketData) -> str:
    direction = "increased" if market.change_pct >= 0 else "decreased"
    headlines_block = (
        "\n".join(f"- {h}" for h in market.news_headlines)
        or "- No recent headlines available."
    )
    return f"""
You are writing one sentence for a professional client email in {language}.

Market data for {stock_name} ({market.ticker}):
- Change: {market.change_pct:+.1f}%
- Direction: {direction}

Recent news headlines:
{headlines_block}

Task:
Write exactly ONE sentence that states the percentage change and gives the most
likely reason based on the headlines above. If the headlines do not clearly explain
the move, attribute it to general market or sector dynamics.

Rules:
- State only the percentage move, do NOT include absolute price figures.
- Write in {language}, professional advisory tone, no hype.
- The sentence must work as a standalone paragraph in the email.

Return ONLY valid JSON:
{{"move_explanation": "one sentence"}}
""".strip()


def build_earnings_prompt(
    stock_name: str, language: str, earnings: EarningsData
) -> str:
    direction = "increased" if earnings.change_pct >= 0 else "decreased"
    headlines_block = (
        "\n".join(f"- {h}" for h in earnings.news_headlines)
        or "- No recent headlines available."
    )

    eps_block = ""
    if earnings.reported_eps is not None:
        eps_block += f"- Reported EPS: {earnings.reported_eps}\n"
    if earnings.estimated_eps is not None:
        eps_block += f"- Estimated EPS: {earnings.estimated_eps}\n"
    if earnings.eps_surprise_pct is not None:
        eps_block += f"- EPS surprise: {earnings.eps_surprise_pct:+.1f}%\n"
    if earnings.revenue is not None:
        eps_block += f"- Revenue: {earnings.currency} {earnings.revenue:,.0f}\n"
    if not eps_block:
        eps_block = "- No detailed earnings metrics available.\n"

    return f"""
You are writing one sentence for a professional client email in {language}.

Latest earnings data for {stock_name} ({earnings.ticker}):
{eps_block.rstrip()}

Market reaction:
- Change: {earnings.change_pct:+.1f}%
- Direction: {direction}

Recent news headlines:
{headlines_block}

Task:
Write exactly ONE sentence that summarises the latest earnings results (EPS, revenue,
beat/miss) and the market reaction. If specific earnings numbers are not available,
focus on the news headlines and price reaction.

Rules:
- State only the percentage move, do NOT include absolute stock price figures.
- You may cite EPS and revenue numbers from the earnings data above.
- Write in {language}, professional advisory tone, no hype.
- The sentence must work as a standalone paragraph in the email.

Return ONLY valid JSON:
{{"move_explanation": "one sentence"}}
""".strip()


def build_sell_prompt(stock_name: str, language: str, factsheet_text: str) -> str:
    return f"""
You are writing a financial analysis for a professional client email in {language}.

Task:
Analyze the factsheet below for {stock_name}.
Extract the key data points that explain why this stock warrants a SELL or weak
recommendation. Focus on: Morningstar / analyst rating, star rating, fair value
estimate vs current price, valuation concerns, risk factors, and investment weaknesses.
Write in professional, client-ready language without hype.

Important PDF encoding note:
In Morningstar factsheets the star rating is often encoded as repeated letter Q.
QQQQQ = 5 stars, QQQQ = 4 stars, QQQ = 3 stars, QQ = 2 stars, Q = 1 star.
Always convert these to the numeric star count (e.g. write "2-star" not "QQ").
Never output the raw Q symbols in your response.

Output requirements:
- Return ONLY valid JSON, no markdown, no extra text.
- Use exactly this schema:
{{
  "analyst_rating": "the recommendation rating stated in the factsheet (e.g. Sell, Reduce, Hold, Accumulate, Buy) or 'Not specified' if absent",
  "morningstar_stars": <integer 1-5 if found in factsheet, otherwise null>,
  "sell_paragraph": "2-4 professional sentences explaining why this stock should be sold. Reference the factsheet rating, star rating, and valuation framework. Every claim must be grounded in the factsheet text.",
  "key_concerns": ["concern 1", "concern 2", "concern 3"]
}}
- Extract ratings and metrics directly from the factsheet. Do not invent facts.
- If the factsheet does not contain a specific metric, acknowledge what data is available instead.
- IMPORTANT: When referencing prices or fair values from the factsheet, always qualify
  them as "the factsheet's figures" or "based on the factsheet's valuation framework".
  Do NOT present factsheet prices as live/current market prices — they may be outdated.
  For example write "Based on the factsheet's valuation framework, the stock appears
  overvalued" rather than "The current price of $X is above...".
- Ensure every sentence has proper spacing between all words.

Factsheet text:
\"\"\"{factsheet_text}\"\"\"
""".strip()


def build_buy_prompt(stock_name: str, language: str, factsheet_text: str) -> str:
    return f"""
You are writing a financial analysis for a professional client email in {language}.

Task:
Analyze the factsheet below for {stock_name}.
Extract the key data points that explain why this stock is a BUY or strong
recommendation as a replacement position. Focus on: Morningstar / analyst rating,
star rating, fair value estimate vs current price, valuation upside, investment
strengths, quality metrics, and growth potential.
Write in professional, client-ready language without hype.

Important PDF encoding note:
In Morningstar factsheets the star rating is often encoded as repeated letter Q.
QQQQQ = 5 stars, QQQQ = 4 stars, QQQ = 3 stars, QQ = 2 stars, Q = 1 star.
Always convert these to the numeric star count (e.g. write "4-star" not "QQQQ").
Never output the raw Q symbols in your response.

Output requirements:
- Return ONLY valid JSON, no markdown, no extra text.
- Use exactly this schema:
{{
  "analyst_rating": "the recommendation rating stated in the factsheet (e.g. Buy, Accumulate, Hold) or 'Not specified' if absent",
  "morningstar_stars": <integer 1-5 if found in factsheet, otherwise null>,
  "buy_paragraph": "2-4 professional sentences explaining why this stock should be bought. Reference the factsheet rating, star rating, and valuation framework. Every claim must be grounded in the factsheet text.",
  "portfolio_fit": "1-2 sentences explaining how this stock improves portfolio positioning and diversification",
  "key_strengths": ["strength 1", "strength 2", "strength 3"]
}}
- Extract ratings and metrics directly from the factsheet. Do not invent facts.
- If the factsheet does not contain a specific metric, acknowledge what data is available instead.
- IMPORTANT: When referencing prices or fair values from the factsheet, always qualify
  them as "the factsheet's figures" or "based on the factsheet's valuation framework".
  Do NOT present factsheet prices as live/current market prices — they may be outdated.
  For example write "Based on the factsheet's valuation, the stock trades below fair
  value" rather than "The current price of $X is below...".
- Ensure every sentence has proper spacing between all words.

Factsheet text:
\"\"\"{factsheet_text}\"\"\"
""".strip()


def build_hold_prompt(stock_name: str, language: str, factsheet_text: str) -> str:
    return f"""
You are writing a financial analysis for a professional client email in {language}.

Task:
Analyze the factsheet below for {stock_name}.
Extract the key data points that support a HOLD recommendation despite recent price
changes. Focus on: Morningstar / analyst rating, star rating, fair value estimate vs
current price, balanced investment case, and factors to monitor.
Write in professional, client-ready language without hype.

Important PDF encoding note:
In Morningstar factsheets the star rating is often encoded as repeated letter Q.
QQQQQ = 5 stars, QQQQ = 4 stars, QQQ = 3 stars, QQ = 2 stars, Q = 1 star.
Always convert these to the numeric star count (e.g. write "3-star" not "QQQ").
Never output the raw Q symbols in your response.

Output requirements:
- Return ONLY valid JSON, no markdown, no extra text.
- Use exactly this schema:
{{
  "analyst_rating": "the recommendation rating stated in the factsheet or 'Not specified' if absent",
  "morningstar_stars": <integer 1-5 if found in factsheet, otherwise null>,
  "hold_paragraph": "2-4 professional sentences explaining why this stock should be held. Reference the factsheet rating, star rating, and valuation framework. Every claim must be grounded in the factsheet text.",
  "watchpoint": "1-2 sentences about what to monitor going forward"
}}
- Extract ratings and metrics directly from the factsheet. Do not invent facts.
- If the factsheet does not contain a specific metric, acknowledge what data is available instead.
- IMPORTANT: When referencing prices or fair values from the factsheet, always qualify
  them as "the factsheet's figures" or "based on the factsheet's valuation framework".
  Do NOT present factsheet prices as live/current market prices — they may be outdated.
- Ensure every sentence has proper spacing between all words.

Factsheet text:
\"\"\"{factsheet_text}\"\"\"
""".strip()


# ── JSON helpers ────────────────────────────────────────────────────


def _extract_json_from_model_text(text: str) -> dict:
    candidate = text.strip()

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("Model did not return valid JSON.")
        return json.loads(candidate[start : end + 1])


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


# ── Payload validators ──────────────────────────────────────────────


def _validate_move_payload(payload: dict) -> dict:
    explanation = _clean(str(payload.get("move_explanation", "")))
    if not explanation:
        raise ValueError("Missing move_explanation in model response.")
    return {"move_explanation": explanation}


def _validate_sell_payload(payload: dict) -> dict:
    paragraph = _clean(str(payload.get("sell_paragraph", "")))
    if not paragraph:
        raise ValueError("Missing sell_paragraph in model response.")
    concerns = [
        _clean(str(c))
        for c in (payload.get("key_concerns") or [])
        if _clean(str(c))
    ]
    if len(concerns) < 2:
        raise ValueError("Expected at least 2 key_concerns.")
    return {
        "analyst_rating": _clean(str(payload.get("analyst_rating", "Not specified"))),
        "morningstar_stars": payload.get("morningstar_stars"),
        "sell_paragraph": paragraph,
        "key_concerns": concerns[:4],
    }


def _validate_buy_payload(payload: dict) -> dict:
    paragraph = _clean(str(payload.get("buy_paragraph", "")))
    if not paragraph:
        raise ValueError("Missing buy_paragraph in model response.")
    fit = _clean(str(payload.get("portfolio_fit", "")))
    if not fit:
        raise ValueError("Missing portfolio_fit in model response.")
    strengths = [
        _clean(str(s))
        for s in (payload.get("key_strengths") or [])
        if _clean(str(s))
    ]
    if len(strengths) < 2:
        raise ValueError("Expected at least 2 key_strengths.")
    return {
        "analyst_rating": _clean(str(payload.get("analyst_rating", "Not specified"))),
        "morningstar_stars": payload.get("morningstar_stars"),
        "buy_paragraph": paragraph,
        "portfolio_fit": fit,
        "key_strengths": strengths[:4],
    }


def _validate_hold_payload(payload: dict) -> dict:
    paragraph = _clean(str(payload.get("hold_paragraph", "")))
    if not paragraph:
        raise ValueError("Missing hold_paragraph in model response.")
    watchpoint = _clean(str(payload.get("watchpoint", "")))
    if not watchpoint:
        raise ValueError("Missing watchpoint in model response.")
    return {
        "analyst_rating": _clean(str(payload.get("analyst_rating", "Not specified"))),
        "morningstar_stars": payload.get("morningstar_stars"),
        "hold_paragraph": paragraph,
        "watchpoint": watchpoint,
    }


# ── OpenAI caller ───────────────────────────────────────────────────


ANALYST_SYSTEM = (
    "You are a careful equity analyst writer. "
    "Extract data only from the provided factsheet text. "
    "Never invent facts or metrics. Follow the output schema exactly."
)

MOVE_SYSTEM = (
    "You are a financial market commentator. "
    "Write one factual sentence about a stock price move. "
    "Follow the output schema exactly."
)

EARNINGS_SYSTEM = (
    "You are a financial market commentator. "
    "Write one factual sentence summarising the latest earnings results "
    "and market reaction. Follow the output schema exactly."
)


def call_openai(
    client: OpenAI,
    prompt: str,
    system_message: str,
    validator: Callable[[dict], dict],
) -> dict:
    response = client.responses.create(
        model=config.OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )

    text = (response.output_text or "").strip()
    if not text:
        raise RuntimeError("OpenAI response was empty.")
    payload = _extract_json_from_model_text(text)
    return validator(payload)


# ── Email builders ──────────────────────────────────────────────────


def _stars_label(stars: int | None) -> str:
    if stars is None:
        return ""
    return f" and a {stars}-star Morningstar rating"


def cleanup_email_text(text: str) -> str:
    """Fix common LLM spacing issues before saving the email."""
    # Insert space between a lowercase letter immediately followed by an uppercase letter
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    # Insert space after a comma/period/semicolon not followed by a space or newline
    text = re.sub(r"([,;])(?=[^\s])", r"\1 ", text)
    text = re.sub(r"(\.)(?=[A-Z])", r". ", text)
    # Collapse multiple spaces (but not newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def build_sell_buy_email(
    stock_name: str,
    isin: str,
    replacement_name: str,
    replacement_isin: str,
    sell_data: dict,
    buy_data: dict,
    move_explanation: str | None,
) -> str:
    intro = (
        f"Your holding {stock_name} ({isin}) in account ACCOUNT_NAME "
        f"is currently recommended for sale."
    )

    if move_explanation:
        move_paragraph = move_explanation
        if not move_paragraph.rstrip().endswith("."):
            move_paragraph = move_paragraph.rstrip() + "."
    else:
        move_paragraph = ""

    sell_rating = sell_data["analyst_rating"]
    sell_stars = _stars_label(sell_data.get("morningstar_stars"))
    sell_section = (
        f"Despite the recent move, we recommend selling {stock_name}. "
        f"The {stock_name} factsheet shows a {sell_rating} rating{sell_stars}, "
        f"which indicates limited attractiveness at the current level. "
        f"{sell_data['sell_paragraph']}"
    )

    buy_rating = buy_data["analyst_rating"]
    buy_stars = _stars_label(buy_data.get("morningstar_stars"))
    buy_section = (
        f"We recommend switching into {replacement_name} ({replacement_isin}). "
        f"The {replacement_name} factsheet shows a {buy_rating} rating{buy_stars}. "
        f"{buy_data['buy_paragraph']}"
    )

    parts = [f"Dear CLIENT_NAME,", "", intro]
    if move_paragraph:
        parts += ["", move_paragraph]
    parts += [
        "",
        sell_section,
        "",
        buy_section,
        "",
        buy_data["portfolio_fit"],
        "",
        f"Bottom line: We recommend selling {stock_name} and switching into "
        f"{replacement_name}.",
        "",
        "Do not hesitate to reach out to your CRO if you have any questions.",
        "",
        "Kind regards,",
        "CRO_NAME",
    ]

    return cleanup_email_text("\n".join(parts) + "\n")


def build_hold_email(
    stock_name: str,
    isin: str,
    hold_data: dict,
    move_explanation: str | None,
) -> str:
    intro = (
        f"Your holding {stock_name} ({isin}) in account ACCOUNT_NAME "
        f"is currently recommended as HOLD."
    )

    if move_explanation:
        move_paragraph = move_explanation
        if not move_paragraph.rstrip().endswith("."):
            move_paragraph = move_paragraph.rstrip() + "."
    else:
        move_paragraph = ""

    hold_rating = hold_data["analyst_rating"]
    hold_stars = _stars_label(hold_data.get("morningstar_stars"))

    hold_section = (
        f"We continue to recommend holding {stock_name}. "
        f"The factsheet shows a {hold_rating} rating{hold_stars}. "
        f"{hold_data['hold_paragraph']}"
    )

    parts = [f"Dear CLIENT_NAME,", "", intro]
    if move_paragraph:
        parts += ["", move_paragraph]
    parts += [
        "",
        hold_section,
        "",
        f"Key watchpoint: {hold_data['watchpoint']}",
        "",
        "Do not hesitate to reach out to your CRO if you have any questions.",
        "",
        "Kind regards,",
        "CRO_NAME",
    ]

    return cleanup_email_text("\n".join(parts) + "\n")


# ── File helpers ────────────────────────────────────────────────────


def sanitize_for_filename(value: str) -> str:
    normalized = normalize_text(value).replace(" ", "_")
    return normalized[:40] if normalized else "unknown"


def save_email(output_dir: Path, filename: str, body: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(body, encoding="utf-8")
    return path


# ── CLI input ───────────────────────────────────────────────────────


def collect_inputs() -> UserInputs:
    stock_name = input("Enter company name: ").strip()
    isin = input("Enter ISIN: ").strip()
    ticker = input("Enter ticker for market data (e.g. ABBN.SW): ").strip() or None
    raw_reason = input("Enter reason (Move / Earnings): ").strip()
    language = input("Enter email language (free text): ").strip()
    raw_action = input("Enter recommendation (Hold / Sell & Buy): ").strip()

    if not stock_name:
        raise ValueError("Company name cannot be empty.")
    if not isin:
        raise ValueError("ISIN cannot be empty.")
    if not language:
        raise ValueError("Language cannot be empty.")

    reason = parse_reason(raw_reason)
    action = parse_action(raw_action)

    replacement_stock = None
    replacement_isin = None
    replacement_ticker = None
    if action == "sell_buy":
        replacement_stock = input("Enter target company name: ").strip()
        replacement_isin = input("Enter target ISIN: ").strip()
        replacement_ticker = (
            input("Enter target ticker for market data: ").strip() or None
        )
        if not replacement_stock:
            raise ValueError("Target company name cannot be empty for Sell & Buy.")
        if not replacement_isin:
            raise ValueError("Target ISIN cannot be empty for Sell & Buy.")

    return UserInputs(
        stock_name=stock_name,
        isin=isin,
        action=action,
        reason=reason,
        language=language,
        ticker=ticker,
        replacement_stock=replacement_stock,
        replacement_isin=replacement_isin,
        replacement_ticker=replacement_ticker,
    )


def validate_config() -> tuple[Path, Path]:
    if not config.OPENAI_API_KEY or "REPLACE_WITH" in config.OPENAI_API_KEY:
        raise ValueError("Set OPENAI_API_KEY in config.py before running.")
    if not config.FACTSHEETS_DIR or "REPLACE_WITH" in config.FACTSHEETS_DIR:
        raise ValueError("Set FACTSHEETS_DIR in config.py before running.")

    factsheet_dir = Path(config.FACTSHEETS_DIR).expanduser()
    if not factsheet_dir.exists() or not factsheet_dir.is_dir():
        raise FileNotFoundError(
            f"FACTSHEETS_DIR does not exist or is not a folder: {factsheet_dir}"
        )

    output_dir = Path(config.OUTPUT_DIR).expanduser()
    return factsheet_dir, output_dir


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    factsheet_dir, output_dir = validate_config()
    user_input = collect_inputs()

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    print(f"\n[INFO] Company: {user_input.stock_name}")
    print(f"[INFO] ISIN: {user_input.isin}")
    print(f"[INFO] Reason: {user_input.reason}")
    print(f"[INFO] Action: {user_input.action}")

    # ── Fetch live data based on reason ──
    move_explanation: str | None = None

    if user_input.ticker:
        print(f"[INFO] Fetching {'earnings' if user_input.reason == 'earnings' else 'market'} data for {user_input.ticker}...")

        if user_input.reason == "earnings":
            earnings = fetch_earnings_data(user_input.ticker)
            if earnings:
                direction = "up" if earnings.change_pct >= 0 else "down"
                print(
                    f"[INFO] {earnings.company_name}: {earnings.currency} "
                    f"{earnings.last_price} ({earnings.change_pct:+.2f}% {direction})"
                )
                earnings_prompt = build_earnings_prompt(
                    user_input.stock_name, user_input.language, earnings
                )
                move_data = call_openai(
                    client, earnings_prompt, EARNINGS_SYSTEM, _validate_move_payload
                )
                move_explanation = move_data["move_explanation"]
                print("[INFO] Earnings explanation generated.")
            else:
                print("[INFO] Earnings data unavailable; email will omit earnings context.")
        else:
            market = fetch_market_data(user_input.ticker)
            if market:
                direction = "up" if market.change_pct >= 0 else "down"
                print(
                    f"[INFO] {market.company_name}: {market.currency} "
                    f"{market.last_price} ({market.change_pct:+.2f}% {direction})"
                )
                move_prompt = build_move_prompt(
                    user_input.stock_name, user_input.language, market
                )
                move_data = call_openai(
                    client, move_prompt, MOVE_SYSTEM, _validate_move_payload
                )
                move_explanation = move_data["move_explanation"]
                print("[INFO] Move explanation generated.")
            else:
                print("[INFO] Market data unavailable; email will omit price-move context.")
    else:
        print("[INFO] No ticker provided; skipping live data retrieval.")

    # ── Match and extract primary factsheet ──
    primary_pdf = find_best_factsheet(user_input.stock_name, factsheet_dir)
    primary_text, primary_engine = extract_pdf_text(primary_pdf)
    print(f"[INFO] Matched primary stock to: {primary_pdf.name}")
    print(
        f"[INFO] Text extracted using: {primary_engine} "
        f"({len(primary_text)} characters)"
    )

    if user_input.action == "hold":
        # ── Hold analysis from primary factsheet ──
        hold_prompt = build_hold_prompt(
            user_input.stock_name, user_input.language, primary_text
        )
        hold_data = call_openai(
            client, hold_prompt, ANALYST_SYSTEM, _validate_hold_payload
        )
        print(
            f"[INFO] Hold analysis complete "
            f"(rating: {hold_data['analyst_rating']}, "
            f"stars: {hold_data.get('morningstar_stars', 'N/A')})"
        )

        email_body = build_hold_email(
            user_input.stock_name, user_input.isin, hold_data, move_explanation
        )
    else:
        # ── Sell analysis from primary factsheet ──
        sell_prompt = build_sell_prompt(
            user_input.stock_name, user_input.language, primary_text
        )
        sell_data = call_openai(
            client, sell_prompt, ANALYST_SYSTEM, _validate_sell_payload
        )
        print(
            f"[INFO] Sell analysis complete "
            f"(rating: {sell_data['analyst_rating']}, "
            f"stars: {sell_data.get('morningstar_stars', 'N/A')})"
        )

        # ── Match and extract replacement factsheet ──
        replacement_name = user_input.replacement_stock or ""
        replacement_isin = user_input.replacement_isin or ""
        replacement_pdf = find_best_factsheet(replacement_name, factsheet_dir)
        replacement_text, replacement_engine = extract_pdf_text(replacement_pdf)
        print(f"[INFO] Matched replacement stock to: {replacement_pdf.name}")
        print(
            f"[INFO] Text extracted using: {replacement_engine} "
            f"({len(replacement_text)} characters)"
        )

        # ── Buy analysis from replacement factsheet ──
        buy_prompt = build_buy_prompt(
            replacement_name, user_input.language, replacement_text
        )
        buy_data = call_openai(
            client, buy_prompt, ANALYST_SYSTEM, _validate_buy_payload
        )
        print(
            f"[INFO] Buy analysis complete "
            f"(rating: {buy_data['analyst_rating']}, "
            f"stars: {buy_data.get('morningstar_stars', 'N/A')})"
        )

        email_body = build_sell_buy_email(
            user_input.stock_name,
            user_input.isin,
            replacement_name,
            replacement_isin,
            sell_data,
            buy_data,
            move_explanation,
        )

    # ── Save ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    action_tag = "hold" if user_input.action == "hold" else "sell_buy"
    first_tag = sanitize_for_filename(user_input.stock_name)

    if user_input.action == "hold":
        filename = f"{timestamp}_{action_tag}_{first_tag}.txt"
    else:
        second_tag = sanitize_for_filename(
            user_input.replacement_stock or "replacement"
        )
        filename = f"{timestamp}_{action_tag}_{first_tag}_to_{second_tag}.txt"

    output_file = save_email(output_dir, filename, email_body)

    print("\n===== GENERATED EMAIL =====\n")
    print(email_body)
    print(f"[INFO] Email saved to: {output_file.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
