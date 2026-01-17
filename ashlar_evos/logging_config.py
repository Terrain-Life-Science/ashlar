"""
Logging configuration for registration pipeline.

Provides structured logging setup with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[Path] = None,
    verbose: bool = True
) -> logging.Logger:
    """
    Set up structured logging for the registration pipeline.
    
    Parameters
    ----------
    log_level : str
        Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' (default: 'INFO')
    log_file : Path, optional
        Path to log file (if None, only console logging)
    verbose : bool
        If True, also log to console (default: True)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger('ashlar_evos')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)  # More detailed in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


@contextmanager
def phase_logging(logger: logging.Logger, phase_name: str):
    """
    Context manager for logging a pipeline phase.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    phase_name : str
        Name of the phase being logged
        
    Example
    -------
    >>> logger = setup_logging()
    >>> with phase_logging(logger, 'Coarse Alignment'):
    ...     # Do work
    ...     pass
    """
    logger.info(f"Starting phase: {phase_name}")
    try:
        yield
        logger.info(f"Completed phase: {phase_name}")
    except Exception as e:
        logger.error(f"Phase {phase_name} failed: {e}", exc_info=True)
        raise


def get_logger(name: str = 'ashlar_evos') -> logging.Logger:
    """
    Get a logger instance.
    
    Parameters
    ----------
    name : str
        Logger name (default: 'ashlar_evos')
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
