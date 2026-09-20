import logging
import sys

from ragforge.core.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configures structured standard application logging."""
    log_format = (
        f'{{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        f'"service": "{settings.app_name}", "env": "{settings.env}", '
        f'"logger": "%(name)s", "message": "%(message)s"}}'
    )

    numeric_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Reconfigure or attach single handler to avoid duplicate lines
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(handler)
    else:
        for existing_handler in root_logger.handlers:
            existing_handler.setLevel(numeric_level)
            existing_handler.setFormatter(logging.Formatter(log_format))
