import logging

# Define TRACE log level
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

# Extend Logger with trace and error methods
def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kws)

def error(self, message, *args, **kws):
    if self.isEnabledFor(logging.ERROR):
        self._log(logging.ERROR, message, args, **kws)

logging.Logger.trace = trace
logging.Logger.error = error


def get_logger(name: str, level: int = logging.ERROR) -> logging.Logger:
    """
    Create and configure a logger with the given name and level.
    The logger uses a StreamHandler with a simple formatter.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
