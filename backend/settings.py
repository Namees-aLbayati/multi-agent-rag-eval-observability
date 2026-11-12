"""Server-only configuration. Never return credentials to the browser."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')

# Migration convenience for the existing notebook, without copying its secrets.
legacy = ROOT.parent / 'config.json'
if legacy.exists():
    config = json.loads(legacy.read_text())
    aliases = {
        'OPENAI_API_KEY': 'OPENAI_API_KEY',
        'OPENAI_API_BASE': 'OPENAI_BASE_URL',
        'LANGCHAIN_API_KEY': 'LANGSMITH_API_KEY',
        'LANGCHAIN_PROJECT': 'LANGSMITH_PROJECT',
    }
    for old, new in aliases.items():
        value = config.get(old)
        if value and not os.getenv(new):
            os.environ[new] = str(value)

os.environ.setdefault('LANGSMITH_PROJECT', 'tenantflow-langgraph')
# Tracing is opt-in, including when legacy config enables tracing.
os.environ.setdefault('LANGSMITH_TRACING', 'false')


def tracing_enabled():
    return os.getenv('LANGSMITH_TRACING', '').lower() == 'true' and bool(os.getenv('LANGSMITH_API_KEY'))


def model(validation=False):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv('VALIDATION_MODEL' if validation else 'OPENAI_MODEL', 'gpt-4o' if validation else 'gpt-4o-mini'),
        temperature=0,
        timeout=60,
        max_retries=1,
    )
