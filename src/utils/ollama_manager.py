import logging
import shutil
import subprocess
import urllib.request

logger = logging.getLogger(__name__)


def _unload_via_http(base_url: str, model_name: str) -> bool:
    """Attempt to unload the model through Ollama's HTTP API."""
    url = f"{base_url.rstrip('/')}/api/ps/{model_name}"
    request = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=5):
            logger.info("Unloaded model %s via HTTP API.", model_name)
            return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("HTTP unload failed for %s: %s", model_name, exc)
        return False


def _unload_via_cli(model_name: str) -> None:
    """Fallback to Ollama CLI if available."""
    command = shutil.which("ollama")
    if not command:
        logger.warning("Ollama CLI not found; cannot unload model %s.", model_name)
        return

    try:
        subprocess.run(
            [command, "stop", model_name],
            check=False,
            capture_output=True,
            text=True,
        )
        logger.info("Executed 'ollama stop %s'.", model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to unload model %s via CLI: %s", model_name, exc)


def unload_model(base_url: str, model_name: str) -> None:
    """
    Release GPU memory by stopping the model process.

    The function tries HTTP first and falls back to the Ollama CLI.
    """

    if not _unload_via_http(base_url, model_name):
        _unload_via_cli(model_name)
