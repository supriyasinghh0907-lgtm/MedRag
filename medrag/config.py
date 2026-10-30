"""All settings come from environment variables (set them as Secrets on Hugging Face).
Never paste an API key into this file."""
import os

# Any OpenAI-compatible endpoint works. Default is Groq's free tier.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL_FAST = os.getenv("LLM_MODEL_FAST", "openai/gpt-oss-20b")      # Agent 1 + classifier
LLM_MODEL_STRONG = os.getenv("LLM_MODEL_STRONG", "openai/gpt-oss-120b")  # Agent 2 validator

EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_PDFS = 10
MAX_PDF_MB = 15
