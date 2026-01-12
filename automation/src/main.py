import argparse
from pathlib import Path
import automation as automation_module
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Small automation tool: validate inputs and produce summary")
    p.add_argument("input", help="Path to input folder")
    p.add_argument("-o", "--output", help="Path to output folder (created if missing)", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve() if args.output else (input_dir / "output")

    ok, log_path = automation_module.run(input_dir, output_dir)
    if not ok:
        print(f"ERROR: Automation failed. See log: {log_path}")
        sys.exit(2)

    print(f"SUCCESS: Automation completed. See output at: {output_dir}")


if __name__ == "__main__":
    main()
