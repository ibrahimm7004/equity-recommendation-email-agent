"""
Project configuration.

Update this file before running the tool.
"""

# OpenAI API key used by the script.
OPENAI_API_KEY = ""

# Folder that contains your local equity factsheet PDFs.
# Example: r"C:\Users\yourname\Documents\equity_factsheets"
FACTSHEETS_DIR = "factsheets"

# Model used for email-summary generation.
OPENAI_MODEL = "gpt-4o-mini"

# Where generated emails are saved (relative to project root unless absolute path).
OUTPUT_DIR = "output_emails"

# Minimum score required for a PDF name match.
MIN_MATCH_SCORE = 32.0

# Number of close match suggestions shown when no file is matched.
MATCH_SUGGESTIONS = 5
