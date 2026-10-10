# app/core/logging_config.py
import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name: str, console: bool | None = None, logfile: str = "/workspace/logs/visionassist.log") -> logging.Logger:
    """
    Factory for consistent logger setup across modules.

    Parameters
    - name: logger name (usually __name__)
    - console: explicit override to enable/disable StreamHandler. If None, uses
      the `LOG_CONSOLE` env var (default True).
    - logfile: path to write rotating file logs.
    """
    logger = logging.getLogger(name)
    # Prevent adding handlers multiple times if get_logger() called repeatedly
    if getattr(logger, "_configured", False):
        return logger

    # Determine logging level
    level = logging.DEBUG if os.getenv("DEBUG_MODE") == "True" else logging.INFO
    logger.setLevel(level)

    # Decide whether to attach a console handler
    if console is None:
        console = os.getenv("LOG_CONSOLE", "True").lower() in ("1", "true", "yes")

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(ch)

    # Ensure log directory exists for file handler
    try:
        log_dir = os.path.dirname(logfile)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # If we cannot create the dir, continue without file handler
        logger.warning("Could not create log directory for '%s'", logfile)
    else:
        fh = RotatingFileHandler(logfile, maxBytes=10 * 1024 * 1024, backupCount=5)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        ))
        logger.addHandler(fh)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    logger._configured = True
    return logger