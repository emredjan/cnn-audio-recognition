from loguru import logger
import sys

def get_logger():

    logger.remove()
    logger.add(sys.stderr, colorize=True, format="<green>{time}</green> <level>{message}</level>")

    return logger
