import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from automation.src.automation import DataValidator

def main():
    parser = argparse.ArgumentParser(
        description="Validate input folder for Movie Recommendation System"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to input folder containing data to validate"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output folder (default: creates 'output' in input folder)"
    )
    
    args = parser.parse_args()
    
    # Validate input folder exists
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    # Set output folder
    if args.output:
        output_folder = args.output
    else:
        output_folder = os.path.join(args.input_folder, "output")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Run validation
    validator = DataValidator()
    success = validator.validate(args.input_folder, output_folder)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()