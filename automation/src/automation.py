import json
import os
from pathlib import Path
import logging

REQUIRED_FILES = ["metadata.json", "data.csv"]
REQUIRED_DIRS = ["assets"]


def validate_input(input_dir: Path):
    """Validate that required files and folders exist and are non-empty.

    Returns: (valid: bool, errors: list[str])
    """
    errors = []
    if not input_dir.exists():
        errors.append(f"Input folder does not exist: {input_dir}")
        return False, errors

    for fname in REQUIRED_FILES:
        fpath = input_dir / fname
        if not fpath.exists():
            errors.append(f"Missing required file: {fname}")
        else:
            try:
                if fpath.stat().st_size == 0:
                    errors.append(f"Required file is empty: {fname}")
            except OSError as e:
                errors.append(f"Unable to access file {fname}: {e}")

    for dname in REQUIRED_DIRS:
        dpath = input_dir / dname
        if not dpath.exists():
            errors.append(f"Missing required directory: {dname}")
        else:
            # must contain at least one file
            files = [p for p in dpath.iterdir() if p.is_file()]
            if len(files) == 0:
                errors.append(f"Required directory is empty: {dname}")

    return len(errors) == 0, errors


def summarize(input_dir: Path):
    """Produce a summary dict with counts and sizes."""
    summary = {}

    # metadata
    md_path = input_dir / "metadata.json"
    try:
        with md_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        metadata = None
    summary["metadata_present"] = metadata is not None
    summary["metadata_keys"] = list(metadata.keys()) if metadata else []

    # data.csv
    data_path = input_dir / "data.csv"
    data_lines = 0
    data_size = 0
    if data_path.exists():
        try:
            with data_path.open("r", encoding="utf-8", errors="replace") as f:
                for _ in f:
                    data_lines += 1
            data_size = data_path.stat().st_size
        except Exception:
            pass
    summary["data_lines"] = data_lines
    summary["data_size_bytes"] = data_size

    # assets
    assets_path = input_dir / "assets"
    assets_count = 0
    assets_total_size = 0
    assets_listing = []
    if assets_path.exists() and assets_path.is_dir():
        for p in assets_path.rglob("*"):
            if p.is_file():
                assets_count += 1
                try:
                    assets_total_size += p.stat().st_size
                except OSError:
                    pass
                assets_listing.append(str(p.relative_to(input_dir)))
    summary["assets_count"] = assets_count
    summary["assets_total_size_bytes"] = assets_total_size
    summary["assets_listing"] = assets_listing

    # totals
    total_files = 0
    total_bytes = 0
    for p in input_dir.rglob("*"):
        if p.is_file():
            total_files += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
    summary["total_files"] = total_files
    summary["total_bytes"] = total_bytes

    return summary


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "automation.log"

    logger = logging.getLogger("automation")
    logger.setLevel(logging.DEBUG)

    # remove previous handlers
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(ch)

    return logger, log_path


def run(input_dir: Path, output_dir: Path):
    logger, log_path = setup_logging(output_dir)
    logger.info(f"Starting automation. input={input_dir} output={output_dir}")

    valid, errors = validate_input(input_dir)
    if not valid:
        logger.error("Validation failed. See error list.")
        for e in errors:
            logger.error(e)
        # write a small error summary file
        err_file = output_dir / "error_summary.json"
        try:
            with err_file.open("w", encoding="utf-8") as f:
                json.dump({"status": "FAILED", "errors": errors}, f, indent=2)
        except Exception:
            logger.exception("Failed to write error summary file")
        return False, log_path

    # produce summary
    summary = summarize(input_dir)
    summary_path = output_dir / "summary.json"
    try:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote summary to {summary_path}")
    except Exception:
        logger.exception("Failed to write summary file")
        return False, log_path

    logger.info("Automation completed successfully.")
    return True, log_path
