from pathlib import Path
from loguru import logger
import sys
from cnn_audio.params import pr

def get_logger():

    logger.remove()
    log_format = (
        "<white>{time:YYYY-MM-DD HH:mm:ss.SSS}</white> | "
        "<level>{level:<9}</level> | "
        "<level>{message}</level>"
    )

    log_file = Path('.') / pr['locations']['runtime_log_file']

    logger.add(sys.stdout, format=log_format, level='DEBUG', colorize=True)
    logger.add(log_file, format=log_format, level='INFO', rotation="100 MB")

    return logger
