"""Shared logging setup: logs go to the console and to logs/<name>.log."""

import logging
import sys

import config

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    config.LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(config.LOG_DIR / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
