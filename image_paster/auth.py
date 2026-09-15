"""Authentication and API key management for Google AI Studio / Gemini."""

from __future__ import annotations
import json
import os
import stat
from pathlib import Path
from typing import Optional, Dict, Any

AUTH_CONFIG_DIR = Path.home() / ".config" / "image-paster"
AUTH_CONFIG_FILE = AUTH_CONFIG_DIR / "auth.json"


def save_auth_token(token: str) -> Path:
    """Save Google AI Studio API token securely in local user configuration.

    Args:
        token: The Google AI Studio API key.

    Returns:
        Path to the saved configuration file.
    """
    token_clean = token.strip()
    if not token_clean:
        raise ValueError("API token cannot be empty.")

    AUTH_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        AUTH_CONFIG_DIR.chmod(stat.S_IRWXU)  # 0o700: user read/write/execute only
    except Exception:
        pass

    data = {"token": token_clean}
    AUTH_CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        AUTH_CONFIG_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600: user read/write only
    except Exception:
        pass

    return AUTH_CONFIG_FILE


def get_stored_token() -> Optional[str]:
    """Retrieve saved API token from local user configuration."""
    if not AUTH_CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(AUTH_CONFIG_FILE.read_text(encoding="utf-8"))
        token = data.get("token")
        return str(token).strip() if token else None
    except Exception:
        return None


def clear_auth_token() -> bool:
    """Remove saved API token from local user configuration."""
    if AUTH_CONFIG_FILE.exists():
        try:
            AUTH_CONFIG_FILE.unlink()
            return True
        except Exception:
            return False
    return False


def get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Resolve the Google AI Studio / Gemini API key from all available sources.

    Order of precedence:
        1. Explicitly passed argument (CLI --api-key or --token)
        2. GOOGLE_API_KEY environment variable
        3. GEMINI_API_KEY environment variable
        4. Stored token in ~/.config/image-paster/auth.json (via image-paster auth login)

    Returns:
        The resolved API key string, or None if not configured.
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    env_google = os.environ.get("GOOGLE_API_KEY")
    if env_google and env_google.strip():
        return env_google.strip()

    env_gemini = os.environ.get("GEMINI_API_KEY")
    if env_gemini and env_gemini.strip():
        return env_gemini.strip()

    stored = get_stored_token()
    if stored and stored.strip():
        return stored.strip()

    return None


def mask_token(token: Optional[str]) -> str:
    """Mask an API key for safe terminal display."""
    if not token:
        return "None"
    t = token.strip()
    if len(t) <= 8:
        return "****"
    return f"{t[:4]}...{t[-4:]}"


def get_auth_status() -> Dict[str, Any]:
    """Inspect the current authentication status and source."""
    if os.environ.get("GOOGLE_API_KEY", "").strip():
        key = os.environ["GOOGLE_API_KEY"].strip()
        return {
            "authenticated": True,
            "source": "environment variable ($GOOGLE_API_KEY)",
            "masked_key": mask_token(key),
            "key": key,
        }

    if os.environ.get("GEMINI_API_KEY", "").strip():
        key = os.environ["GEMINI_API_KEY"].strip()
        return {
            "authenticated": True,
            "source": "environment variable ($GEMINI_API_KEY)",
            "masked_key": mask_token(key),
            "key": key,
        }

    stored = get_stored_token()
    if stored:
        return {
            "authenticated": True,
            "source": f"stored credentials ({AUTH_CONFIG_FILE})",
            "masked_key": mask_token(stored),
            "key": stored,
        }

    return {
        "authenticated": False,
        "source": "none",
        "masked_key": "None",
        "key": None,
    }
