import logging
import sys


class CustomLogger:
    def __init__(self, script_name, log_level: str = "INFO"):
        self.script_name = script_name
        self.log_level = log_level
        self.logger = logging.getLogger(script_name)
        self._configure_logger(self.log_level)

    def _configure_logger(self, log_level: str):
        log_level_constant = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(log_level_constant)
        # Clear existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level_constant)
        formatter = logging.Formatter(f"%(asctime)s - %(levelname)s - {self.script_name} - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
