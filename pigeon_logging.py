import logging as py_logging

# Define TRACE log level
TRACE = 5
py_logging.addLevelName(TRACE, "TRACE")

# Extend Logger with trace and error methods
def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kws)

def error(self, message, *args, **kws):
    if self.isEnabledFor(py_logging.ERROR):
        self._log(py_logging.ERROR, message, args, **kws)

py_logging.Logger.trace = trace
py_logging.Logger.error = error


def get_logger(name: str, level: int = py_logging.ERROR) -> py_logging.Logger:
    """
    Create and configure a logger with the given name and level.
    The logger uses a StreamHandler with a simple formatter.
    """
    logger = py_logging.getLogger(name)
    logger.setLevel(level)
    ch = py_logging.StreamHandler()
    formatter = py_logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
