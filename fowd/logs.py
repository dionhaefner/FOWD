import logging


def setup_file_logger(logfile):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=logfile,
        filemode='w'
    )