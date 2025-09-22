"""
Author: Shiqi (Anya) WANG
Date: 2025/3/8
Description: Logging system module
For creating and configuring loggers
"""

import os
import logging
import time
from pathlib import Path


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger
    
    Parameters:
        name (str): Logger name
        log_file (str, optional): Log file path, None means no file output
        level (int, optional): Log level, default INFO
    
    Returns:
        logging.Logger: Configured logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    

    if logger.hasHandlers():
        logger.handlers.clear()
    

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    

    logger.addHandler(console_handler)
    

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ProgressLogger:
    """Progress logger for long-running task progress reporting"""
    
    def __init__(self, logger, total, description="Processing", update_interval=5):
        """
        Initialize progress logger
        
        Parameters:
            logger (logging.Logger): Logger
            total (int): Total task count
            description (str): Task description
            update_interval (int): Update interval (seconds)
        """
        self.logger = logger
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.start_time = None
        self.last_update_time = None
        self.completed = 0
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.completed = 0
        self.logger.info(f"Starting {self.description}: {self.total} tasks")
    
    def update(self, increment=1, force=False):
        """
        Update progress
        
        Parameters:
            increment (int): Completed task increment
            force (bool): Force log output regardless of time interval
        """
        self.completed += increment
        current_time = time.time()
        

        time_elapsed = current_time - self.last_update_time
        if force or time_elapsed >= self.update_interval:
            percentage = min(100.0, 100.0 * self.completed / self.total)
            elapsed = current_time - self.start_time
            
            if self.completed > 0:
                remaining = elapsed * (self.total - self.completed) / self.completed
                eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
            else:
                eta = "Unknown"
            
            self.logger.info(
                f"{self.description}: {self.completed}/{self.total} "
                f"({percentage:.2f}%), elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}, "
                f"ETA: {eta}"
            )
            self.last_update_time = current_time
    
    def finish(self):
        """Complete task and output summary"""
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"{self.description} completed! Processed {self.completed} tasks, "
            f"total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
        )


def log_memory_usage(logger, prefix=""):
    """
    Log current memory usage
    
    Parameters:
        logger (logging.Logger): Logger
        prefix (str): Log prefix
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        

        rss_gb = memory_info.rss / (1024 ** 3)
        vms_gb = memory_info.vms / (1024 ** 3)
        

        system_memory = psutil.virtual_memory()
        total_gb = system_memory.total / (1024 ** 3)
        available_gb = system_memory.available / (1024 ** 3)
        used_percent = system_memory.percent
        
        logger.info(f"{prefix}Memory usage: RSS={rss_gb:.2f}GB, VMS={vms_gb:.2f}GB | "
                   f"System memory: Total={total_gb:.2f}GB, Available={available_gb:.2f}GB, Used={used_percent:.1f}%")
    
    except ImportError:
        logger.warning("Cannot log memory usage: please install psutil library (pip install psutil)")
    except Exception as e:
        logger.warning(f"Error logging memory usage: {e}")