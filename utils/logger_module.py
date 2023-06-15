import logging
from logging.handlers import RotatingFileHandler

def get_logger(logfile_path:str):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = RotatingFileHandler(logfile_path, maxBytes=1 * 1024 * 1024, backupCount=10)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d -%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger