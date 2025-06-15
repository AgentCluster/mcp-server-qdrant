import logging
import os
import sys

LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"))
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        log_file = os.path.join(LOG_DIR, f"{name.replace('.', '_')}.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Add console handler for stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        # Root logger'a da handler ekle
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            root_logger.addHandler(handler)
            root_logger.addHandler(console_handler)
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    return logger
