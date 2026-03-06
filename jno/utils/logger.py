import logging
from pathlib import Path
from typing import Optional, Union, Tuple


class Logger:
    """
    A flexible logger that can log to file and/or console.
    Now supports multiple independent instances instead of singleton.
    """

    def __init__(self, path: Union[str, Path] = Path("./"), log_print: Tuple[bool, bool] = (True, True), name: str = "log.txt"):
        path = Path(path) if isinstance(path, str) else path
        self.path: Path = path
        self.log_print: Tuple[bool, bool] = log_print
        self.do_log, self.do_print = log_print

        # Generate logger name
        if name is None:
            if path:
                name = f"CustomLogger_{str(path).replace('/', '_')}"
            else:
                import time

                name = f"CustomLogger_{id(self)}_{int(time.time() * 1000000)}"

        self._logger_name = name  # Store for potential re-creation
        self._setup_logger(name)

    def __getstate__(self):
        """Prepare state for pickling - exclude unpicklable objects."""
        state = self.__dict__.copy()
        # Remove the actual logger and handlers (they contain thread locks)
        state["logger"] = None
        state["file_handler"] = None
        state["console_handler"] = None
        # Store the logger name for reconstruction
        state["_logger_name"] = self.logger.name if self.logger else None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling - reconstruct logger."""
        # Extract the logger name before restoring state
        logger_name = state.pop("_logger_name", None)
        self.__dict__.update(state)

        # Reconstruct the logger and handlers
        if logger_name:
            self._setup_logger(logger_name)
        else:
            self.logger = None

    def _setup_logger(self, name: str):
        """Helper to set up logger and handlers (used by __init__ and __setstate__)."""
        log_level = logging.INFO

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.file_handler = None
        self.console_handler = None

        # Recreate file handler
        if self.do_log and self.path is not None:
            log_file = self.path / "log.txt"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.file_handler = logging.FileHandler(str(log_file))
            self.file_handler.setLevel(log_level)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)

        # Recreate console handler
        if self.do_print:
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(log_level)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            self.console_handler.setFormatter(formatter)
            self.logger.addHandler(self.console_handler)

    def _output(self, message: str, level: int = logging.INFO) -> None:
        """Internal method to handle both logging and printing."""
        if self.do_log or self.do_print:
            if level == logging.INFO:
                self.logger.info(message)
            elif level == logging.WARNING:
                self.logger.warning(message)
            elif level == logging.ERROR:
                self.logger.error(message)
            elif level == logging.DEBUG:
                self.logger.debug(message)
            elif level == logging.CRITICAL:
                self.logger.critical(message)

        # If neither logging nor printing, fall back to print
        if not self.do_log and not self.do_print:
            print(message)

    def _output_file_only(self, message: str, level: int = logging.INFO) -> None:
        """Internal method to log ONLY to file, not console."""
        if self.file_handler is None:
            return

        # Temporarily remove console handler
        console_was_attached = False
        if self.console_handler and self.console_handler in self.logger.handlers:
            self.logger.removeHandler(self.console_handler)
            console_was_attached = True

        # Log to file only
        if level == logging.INFO:
            self.logger.info(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.DEBUG:
            self.logger.debug(message)
        elif level == logging.CRITICAL:
            self.logger.critical(message)

        # Re-attach console handler if it was there
        if console_was_attached:
            self.logger.addHandler(self.console_handler)

    def __call__(self, message: str = "", level: int = logging.INFO) -> None:
        self._output(message, level)
        return None

    def info(self, message: str) -> None:
        self._output(message, logging.INFO)

    def warning(self, message: str) -> None:
        self._output(message, logging.WARNING)

    def error(self, message: str) -> None:
        self._output(message, logging.ERROR)

    def debug(self, message: str) -> None:
        self._output(message, logging.DEBUG)

    def critical(self, message: str) -> None:
        self._output(message, logging.CRITICAL)

    def quiet(self, message: str, level: int = logging.INFO) -> None:
        """Log to file only, no console output."""
        self._output_file_only(message, level)

    def close(self):
        """Close all handlers and cleanup."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


_default_logger: Optional[Logger] = None


class PrintFallback:
    """Fallback logger that uses print when real logger isn't needed."""

    def __init__(self):
        self.path: Path = Path("")

    def __call__(self, message: str = "", level: int = logging.INFO) -> None:
        print(message)
        return None

    def info(self, message: str) -> None:
        print(f"INFO: {message}")

    def warning(self, message: str) -> None:
        print(f"WARNING: {message}")

    def error(self, message: str) -> None:
        print(f"ERROR: {message}")

    def debug(self, message: str) -> None:
        print(f"DEBUG:  {message}")

    def critical(self, message: str) -> None:
        print(f"CRITICAL: {message}")

    def quiet(self, message: str, level: int = logging.INFO) -> None:
        """In fallback mode, quiet does nothing."""
        pass

    def close(self):
        """No-op for fallback."""
        pass


def get_logger(path: Optional[Path] = None, log_print: Tuple[bool, bool] = (True, True), use_default: bool = True) -> Union[Logger, PrintFallback]:
    """
    Get a Logger instance.

    :param path: Path to log directory
    :param log_print:  Tuple of (log_to_file, print_to_console)
    :param use_default: If True, returns/creates a shared default logger
                        If False, always creates a new independent logger
    :return: Logger instance or PrintFallback
    """
    global _default_logger

    if use_default:
        # Singleton behavior for backward compatibility
        if _default_logger is None and path is not None:
            _default_logger = Logger(path=path, log_print=log_print, name="DefaultLogger")
        return _default_logger if _default_logger is not None else PrintFallback()
    else:
        # Always create a new independent logger
        if path is not None:
            return Logger(path=path, log_print=log_print)
        else:
            return PrintFallback()


def init_default_logger(path: Path, log_print: Tuple[bool, bool] = (True, True), force: bool = False) -> Logger:
    """Initialize (or rebind) the default logger.

    Args:
        path: Directory where ``log.txt`` will be written.
        log_print: Tuple ``(log_to_file, print_to_console)``.
        force: If ``True``, always recreate the singleton logger for *path*.
    """
    global _default_logger

    path = Path(path)

    if _default_logger is not None:
        current_path = getattr(_default_logger, "path", None)
        current_cfg = getattr(_default_logger, "log_print", None)
        rebind = bool(force)

        if current_path is None:
            rebind = True
        else:
            try:
                rebind = rebind or (Path(current_path).resolve() != path.resolve())
            except Exception:
                rebind = True

        if current_cfg is not None:
            rebind = rebind or (current_cfg != log_print)

        if rebind:
            try:
                _default_logger.close()
            except Exception:
                pass
            _default_logger = None

    if _default_logger is None:
        _default_logger = Logger(path=path, log_print=log_print, name="DefaultLogger")

    return _default_logger
