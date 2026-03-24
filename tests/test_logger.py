from pathlib import Path

from jno.utils.logger import Logger


def test_logger_reports_log_file_path_once(tmp_path: Path):
    logger = Logger(path=tmp_path, log_print=(True, False), name="test_logger_reports_log_file_path_once")
    logger.close()

    log_file = tmp_path / "log.txt"
    contents = log_file.read_text()

    expected_line = f"Log file: {log_file.resolve()}"
    assert contents.count(expected_line) == 1
