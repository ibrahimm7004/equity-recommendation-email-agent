from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import config


try:
    import pdfplumber
except ImportError:  # pragma: no cover - runtime dependency
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - runtime dependency
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - runtime dependency
    PdfReader = None

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
    action: str
    language: str
    replacement_stock: str | None = None


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
        except Exception as exc:  # pragma: no cover - defensive branch
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


def parse_action(raw_action: str) -> str:
    action = normalize_text(raw_action)
    if action in {"hold", "h"}:
        return "hold"
    if action in {"sell", "sell buy", "sell and buy", "sell buy a new stock", "s"}:
        return "sell_buy"
    if "sell" in action and "buy" in action:
        return "sell_buy"
    raise ValueError("Action must be Hold or Sell & Buy.")


def build_hold_prompt(stock_name: str, language: str, factsheet_text: str) -> str:
    return f"""
You are writing a financial client email summary in {language}.

Task:
Use ONLY the factsheet text below.
Explain why the recommendation is to HOLD {stock_name} after recent price changes.
Write in professional, client-ready language without hype.

Output requirements:
- Return ONLY valid JSON, no markdown, no extra text.
- Use exactly this JSON schema:
{{
  "thesis": "one concise sentence",
  "support_points": ["sentence 1", "sentence 2", "sentence 3"],
  "watchpoint": "one concise sentence"
}}
- Do not invent facts.
- If the factsheet does not support a detail, write a cautious generic statement grounded in available text.

Factsheet text:
\"\"\"{factsheet_text}\"\"\"
""".strip()


def build_sell_prompt(stock_name: str, language: str, factsheet_text: str) -> str:
    return f"""
You are writing a financial client email summary in {language}.

Task:
Use ONLY the factsheet text below.
Explain why the recommendation is to SELL {stock_name} after recent price changes.
Write in professional, client-ready language without hype.

Output requirements:
- Return ONLY valid JSON, no markdown, no extra text.
- Use exactly this JSON schema:
{{
  "thesis": "one concise sentence",
  "support_points": ["sentence 1", "sentence 2", "sentence 3"],
  "execution_note": "one concise sentence"
}}
- Do not invent facts.
- If the factsheet does not support a detail, write a cautious generic statement grounded in available text.

Factsheet text:
\"\"\"{factsheet_text}\"\"\"
""".strip()


def build_buy_prompt(stock_name: str, language: str, factsheet_text: str) -> str:
    return f"""
You are writing a financial client email summary in {language}.

Task:
Use ONLY the factsheet text below.
Explain why the recommendation is to BUY {stock_name} as a replacement position.
Write in professional, client-ready language without hype.

Output requirements:
- Return ONLY valid JSON, no markdown, no extra text.
- Use exactly this JSON schema:
{{
  "thesis": "one concise sentence",
  "support_points": ["sentence 1", "sentence 2", "sentence 3"],
  "portfolio_fit": "one concise sentence"
}}
- Do not invent facts.
- If the factsheet does not support a detail, write a cautious generic statement grounded in available text.

Factsheet text:
\"\"\"{factsheet_text}\"\"\"
""".strip()


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


def _clean_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    return cleaned


def _validate_summary_payload(payload: dict, required_tail_key: str) -> dict:
    thesis = _clean_sentence(str(payload.get("thesis", "")))
    support_points = payload.get("support_points", [])
    tail_value = _clean_sentence(str(payload.get(required_tail_key, "")))

    if not isinstance(support_points, list):
        raise ValueError("support_points must be a list.")

    points = [_clean_sentence(str(item)) for item in support_points if _clean_sentence(str(item))]
    if len(points) < 3:
        raise ValueError("Expected at least 3 support_points.")

    if not thesis:
        raise ValueError("Missing thesis in model response.")
    if not tail_value:
        raise ValueError(f"Missing {required_tail_key} in model response.")

    return {
        "thesis": thesis,
        "support_points": points[:3],
        required_tail_key: tail_value,
    }


def call_openai_summary(client: OpenAI, prompt: str, required_tail_key: str) -> dict:
    response = client.responses.create(
        model=config.OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a careful equity analyst writer. "
                    "Use only provided factsheet text, avoid unsupported claims, "
                    "and follow output format exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    text = (response.output_text or "").strip()
    if not text:
        raise RuntimeError("OpenAI response was empty.")
    payload = _extract_json_from_model_text(text)
    return _validate_summary_payload(payload, required_tail_key)


def render_hold_summary(stock_name: str, payload: dict) -> str:
    points = payload["support_points"]
    return (
        f"Recommendation: Hold {stock_name}\n\n"
        "Investment rationale:\n"
        f"1. {payload['thesis']}\n"
        f"2. {points[0]}\n"
        f"3. {points[1]}\n"
        f"4. {points[2]}\n\n"
        f"Key watchpoint: {payload['watchpoint']}"
    )


def render_sell_summary(stock_name: str, payload: dict) -> str:
    points = payload["support_points"]
    return (
        f"Sell rationale ({stock_name}):\n"
        f"1. {payload['thesis']}\n"
        f"2. {points[0]}\n"
        f"3. {points[1]}\n"
        f"4. {points[2]}\n\n"
        f"Execution note: {payload['execution_note']}"
    )


def render_buy_summary(stock_name: str, payload: dict) -> str:
    points = payload["support_points"]
    return (
        f"Buy rationale ({stock_name}):\n"
        f"1. {payload['thesis']}\n"
        f"2. {points[0]}\n"
        f"3. {points[1]}\n"
        f"4. {points[2]}\n\n"
        f"Portfolio fit: {payload['portfolio_fit']}"
    )


def sanitize_for_filename(value: str) -> str:
    normalized = normalize_text(value).replace(" ", "_")
    return normalized[:40] if normalized else "unknown"


def build_email_text(
    action: str,
    stock_name: str,
    summary_primary: str,
    replacement_stock: str | None = None,
    summary_replacement: str | None = None,
) -> str:
    if action == "hold":
        first_line = (
            f"Your holding {stock_name} (ISIN_NUMBER) in account ACCOUNT_NAME "
            "is currently recommended as HOLD."
        )
        summary_block = summary_primary
    else:
        first_line = (
            f"Your holding {stock_name} (ISIN_NUMBER) in account ACCOUNT_NAME "
            f"is currently recommended to SELL and replace with {replacement_stock}."
        )
        summary_block = (
            f"Recommendation: Sell {stock_name} and buy {replacement_stock}\n\n"
            f"{summary_primary}\n\n"
            f"{summary_replacement or ''}"
        ).strip()

    return (
        "Dear CLIENT_NAME\n\n"
        f"{first_line}\n\n"
        "Here will follow a summary breakdown of *why* we're recommending the action.\n\n"
        f"{summary_block}\n\n"
        "Do not hesitate to reach out to your CRO if you have any questions.\n\n"
        "Kind regards,\n"
        "CRO_NAME\n"
    )


def save_email(output_dir: Path, filename: str, body: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(body, encoding="utf-8")
    return path


def collect_inputs() -> UserInputs:
    stock_name = input("Enter stock name: ").strip()
    raw_action = input("Enter recommendation type (Hold / Sell & Buy): ").strip()
    language = input("Enter email language (free text): ").strip()

    if not stock_name:
        raise ValueError("Stock name cannot be empty.")
    if not language:
        raise ValueError("Language cannot be empty.")

    action = parse_action(raw_action)
    replacement_stock = None
    if action == "sell_buy":
        replacement_stock = input("Enter replacement stock name to buy: ").strip()
        if not replacement_stock:
            raise ValueError("Replacement stock name cannot be empty for Sell & Buy.")

    return UserInputs(
        stock_name=stock_name,
        action=action,
        language=language,
        replacement_stock=replacement_stock,
    )


def validate_config() -> tuple[Path, Path]:
    if not config.OPENAI_API_KEY or "REPLACE_WITH" in config.OPENAI_API_KEY:
        raise ValueError("Set OPENAI_API_KEY in config.py before running.")
    if not config.FACTSHEETS_DIR or "REPLACE_WITH" in config.FACTSHEETS_DIR:
        raise ValueError("Set FACTSHEETS_DIR in config.py before running.")

    factsheet_dir = Path(config.FACTSHEETS_DIR).expanduser()
    if not factsheet_dir.exists() or not factsheet_dir.is_dir():
        raise FileNotFoundError(f"FACTSHEETS_DIR does not exist or is not a folder: {factsheet_dir}")

    output_dir = Path(config.OUTPUT_DIR).expanduser()
    return factsheet_dir, output_dir


def main() -> None:
    factsheet_dir, output_dir = validate_config()
    user_input = collect_inputs()

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    primary_pdf = find_best_factsheet(user_input.stock_name, factsheet_dir)
    primary_text, primary_engine = extract_pdf_text(primary_pdf)

    print(f"[INFO] Matched primary stock to: {primary_pdf.name}")
    print(f"[INFO] Text extracted using: {primary_engine} ({len(primary_text)} characters)")

    if user_input.action == "hold":
        hold_prompt = build_hold_prompt(user_input.stock_name, user_input.language, primary_text)
        hold_payload = call_openai_summary(client, hold_prompt, required_tail_key="watchpoint")
        primary_summary = render_hold_summary(user_input.stock_name, hold_payload)
        replacement_summary = None
        replacement_name = None
    else:
        sell_prompt = build_sell_prompt(user_input.stock_name, user_input.language, primary_text)
        sell_payload = call_openai_summary(client, sell_prompt, required_tail_key="execution_note")
        primary_summary = render_sell_summary(user_input.stock_name, sell_payload)

        replacement_name = user_input.replacement_stock or ""
        replacement_pdf = find_best_factsheet(replacement_name, factsheet_dir)
        replacement_text, replacement_engine = extract_pdf_text(replacement_pdf)

        print(f"[INFO] Matched replacement stock to: {replacement_pdf.name}")
        print(
            f"[INFO] Text extracted using: {replacement_engine} "
            f"({len(replacement_text)} characters)"
        )

        buy_prompt = build_buy_prompt(replacement_name, user_input.language, replacement_text)
        buy_payload = call_openai_summary(client, buy_prompt, required_tail_key="portfolio_fit")
        replacement_summary = render_buy_summary(replacement_name, buy_payload)

    email_body = build_email_text(
        action=user_input.action,
        stock_name=user_input.stock_name,
        summary_primary=primary_summary,
        replacement_stock=replacement_name,
        summary_replacement=replacement_summary,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    action_tag = "hold" if user_input.action == "hold" else "sell_buy"
    first_name_tag = sanitize_for_filename(user_input.stock_name)

    if user_input.action == "hold":
        filename = f"{timestamp}_{action_tag}_{first_name_tag}.txt"
    else:
        second_name_tag = sanitize_for_filename(replacement_name or "replacement")
        filename = f"{timestamp}_{action_tag}_{first_name_tag}_to_{second_name_tag}.txt"

    output_file = save_email(output_dir, filename, email_body)

    print("\n===== GENERATED EMAIL =====\n")
    print(email_body)
    print(f"[INFO] Email saved to: {output_file.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
