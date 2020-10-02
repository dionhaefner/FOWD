"""
logs.py

Logging utilities.
"""

import logging


def setup_file_logger(logfile):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=logfile,
        filemode='w'
    )
    logging.captureWarnings(True)

    # silence third-party loggers
    logging.getLogger('filelock').setLevel('CRITICAL')
