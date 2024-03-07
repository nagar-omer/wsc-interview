import logging

LOGGER_NAME = 'WSC-Interview'

# create logger with 'spam_application'
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(ch)

