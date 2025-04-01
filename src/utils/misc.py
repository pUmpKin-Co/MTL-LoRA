import os
import logging
import datetime


def add_filehandler(logger, output):
    if output.endswith(".txt") or output.endswith(".log"):
        filename = output
    else:
        now = datetime.datetime.now()
        filename = os.path.join(output, now.strftime("log-%m-%d-%H-%M-%S.log"))

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] "
        "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)