import logging
import os

class LoggerSingleton:
    _logger = None

    @classmethod
    def setup_logger(cls, working_dir):
        """Sets up the logger only if it hasn't been initialized yet, using the specified working directory."""
        if cls._logger is None:
            log_dir = os.path.join(working_dir, 'log', 'logs')
            log_file_path = os.path.join(log_dir, 'pipeline_logs.txt')
            os.makedirs(log_dir, exist_ok=True)

            cls._logger = logging.getLogger('pipeline_logs')
            cls._logger.setLevel(logging.INFO)
            cls._logger.propagate = False  # Prevent logging propagation

            # clear existing handlers to prevent duplicate logging
            if cls._logger.hasHandlers():
                cls._logger.handlers.clear()

            # file handler setup
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            cls._logger.addHandler(file_handler)

            # console handler setup
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            cls._logger.addHandler(console_handler)

    @classmethod
    def get_logger(cls):
        """Retrieves the logger instance, ensuring it is initialized."""
        if cls._logger is None:
            raise RuntimeError("Logger has not been set up yet. Call setup_logger() first.")
        return cls._logger
