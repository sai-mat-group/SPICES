"""
I/O utilities for reading/writing screening results and configuring
logging.
"""

import logging


def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for SPICES.

    Parameters
    ----------
    level : int
        Logging level (default ``logging.INFO``).
    log_file : str or None
        If provided, also log to this file.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
