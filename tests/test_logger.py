from pathlib import Path

from jno.utils.logger import Logger


def test_logger_reports_log_file_path_once(tmp_path: Path):
    logger = Logger(
        path=tmp_path,
        log_print=(True, False),
        name="test_logger_reports_log_file_path_once",
    )
    logger.close()

    log_file = tmp_path / "log.log"
    contents = log_file.read_text()

    expected_line = f"Log file: {log_file.resolve()}"
    assert contents.count(expected_line) == 1


def test_jno_logger_is_the_run_logger_and_takes_no_options(tmp_path: Path):
    """jno.logger() hands back the logger jno.setup / a path started, for messages of your own; it has no
    logging options of its own (file and console, always)."""
    import inspect

    import jno
    from jno.utils import logger as _logger_mod

    assert list(inspect.signature(jno.logger).parameters) == ["path"]
    prev = _logger_mod._default_logger
    try:
        log = jno.logger(tmp_path)
        assert jno.logger() is log
        log.info("a message of my own")
        log.close()
        assert "a message of my own" in (tmp_path / "log.log").read_text()
    finally:
        _logger_mod._default_logger = prev
