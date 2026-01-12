Small Automation - Automation & Python Intern - Media Systems

Overview

This small automation validates an input folder for required files and produces a summary and a log file. It fails safely with clear errors when inputs are missing or empty.

What it does

- Validates the presence of `metadata.json`, `data.csv`, and a non-empty `assets/` directory.
- If validation fails: writes `error_summary.json` and `automation.log` in the output folder, prints a clear message, and exits non-zero.
- If validation succeeds: writes `summary.json` and `automation.log` in the output folder.

How to run

From the workspace root run:

```bash
python automation/src/main.py examples/input_valid -o examples/output_valid
```

Replace `examples/input_valid` with your input folder. If `-o` is omitted an `output` folder will be created inside the input folder.

Design decisions (brief)

- Required files: `metadata.json` and `data.csv`. I chose these because many media automations need metadata and tabular manifests.
- Required directory: `assets/` must contain at least one file. This prevents silent success on empty package.
- Logging: human-readable `automation.log` with timestamps; machine-readable `summary.json` or `error_summary.json` for downstream consumption.
- Failure mode: non-zero exit code and an error artifact in the `output` folder. This ensures CI or orchestration can detect failures reliably.

One edge case I considered

- Partially unreadable files (permission errors). The code attempts to stat and read files and will include access errors in validation.

One scaling improvement

- Replace in-process file scanning with a streaming worker and add concurrency (e.g., thread pool) for very large asset trees; push metrics to a monitoring endpoint instead of only writing local files.

Files created

- `automation/src/main.py` - CLI entrypoint
- `automation/src/automation.py` - core logic
- `examples/input_valid/` - sample input data (provided)

