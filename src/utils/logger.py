"""
src/utils/logger.py
===================
Centralised logging configuration for the entire project.
 
Uses `loguru` for rich, structured, coloured log output to both
the terminal and a rotating log file.
 
Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.warning("Low sample count")
    logger.error("File not found")
"""
 
import sys
from pathlib import Path
from loguru import logger as _loguru_logger
 
 
# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
 
LOG_FILE = LOG_DIR / "semen_analysis.log"
 
# Log format for terminal — coloured, human-readable
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)
 
# Log format for file — plain text, machine-parseable
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level: <8} | "
    "{name}:{function}:{line} — {message}"
)
 
 
def setup_logger(level: str = "INFO") -> None:
    """
    Configure loguru sinks (console + rotating file).
 
    Parameters
    ----------
    level : str
        Minimum log level to capture.  Defaults to "INFO".
        Accepts: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL.
    """
    # Remove the default loguru sink so we can add our own
    _loguru_logger.remove()
 
    # ── Console sink ──────────────────────────────────────────
    _loguru_logger.add(
        sys.stdout,
        format=CONSOLE_FORMAT,
        level=level,
        colorize=True,
    )
 
    # ── Rotating file sink (max 10 MB, keep 7 backups) ────────
    _loguru_logger.add(
        str(LOG_FILE),
        format=FILE_FORMAT,
        level="DEBUG",          # always capture full debug to file
        rotation="10 MB",
        retention=7,
        compression="zip",
        enqueue=True,           # thread-safe async writes
    )
 
    _loguru_logger.info("Logger initialised — writing to {}", LOG_FILE)
 
 
def get_logger(name: str = "semen_analysis"):
    """
    Return the shared loguru logger instance bound to *name*.
 
    Parameters
    ----------
    name : str
        Module name — typically pass ``__name__``.
 
    Returns
    -------
    loguru.Logger
        Bound logger instance.
    """
    return _loguru_logger.bind(name=name)
 
 
# ── Auto-initialise on first import ───────────────────────────
setup_logger()