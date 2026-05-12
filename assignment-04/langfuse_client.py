import os

from dotenv import load_dotenv

load_dotenv()

_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
_host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

langfuse = None

if _public_key and _secret_key:
    try:
        from langfuse import Langfuse
        langfuse = Langfuse(
            public_key=_public_key,
            secret_key=_secret_key,
            host=_host,
        )
    except Exception as exc:
        print(f"[Langfuse] Failed to initialise client: {exc}")
else:
    print("[Langfuse] LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set — tracing disabled.")
