# Equity Email AI Tool

Python CLI tool that:
- Takes stock/action/language input
- Matches stock names to local factsheet PDFs
- Extracts PDF text with multiple extraction engines
- Calls OpenAI to generate recommendation summary text
- Builds the final client email in your required format
- Prints it and saves it to `output_emails/`

## 1) Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 2) Configure

Open `config.py` and set:
- `OPENAI_API_KEY`
- `FACTSHEETS_DIR`
- (optional) `OPENAI_MODEL`, `OUTPUT_DIR`, match settings

## 3) Run

```powershell
python equity_email_tool.py
```

You will be prompted for:
- Stock name
- Recommendation type (`Hold` or `Sell & Buy`)
- Email language (free text)
- Replacement stock name (only for `Sell & Buy`)

## Notes

- The tool only uses your local factsheet PDFs as source text.
- If no factsheet is matched confidently, it returns an error with top candidate filenames.
- Generated emails are saved as timestamped `.txt` files in `output_emails/`.
