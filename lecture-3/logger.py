"""Shared logging setup: logs go to the console and to logs/<name>.log."""

import logging
import sys

from config_loader import load_config

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    config = load_config()
    logger.setLevel(config["logging"]["level"])
    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = config["paths"]["log_dir"]
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
