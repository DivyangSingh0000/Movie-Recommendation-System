"""
Core validation logic for Movie Recommendation System data.
"""

import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

class DataValidator:
    """Validates input data structure and content for the recommendation system."""
    
    REQUIRED_FILES = ["metadata.json", "data.csv"]
    REQUIRED_DIRS = ["assets"]
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.validation_results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("DataValidator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def validate(self, input_folder: str, output_folder: str) -> bool:
        """
        Validate the input folder structure and contents.
        
        Args:
            input_folder: Path to input folder
            output_folder: Path to output folder
            
        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info(f"Starting validation of: {input_folder}")
        
        try:
            # Initialize validation results
            self.validation_results = {
                "validation_result": "success",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "input_folder": input_folder,
                "files_validated": {},
                "errors": [],
                "warnings": []
            }
            
            # Validate required files
            for file_name in self.REQUIRED_FILES:
                file_path = os.path.join(input_folder, file_name)
                self._validate_file(file_name, file_path)
            
            # Validate required directories
            for dir_name in self.REQUIRED_DIRS:
                dir_path = os.path.join(input_folder, dir_name)
                self._validate_directory(dir_name, dir_path)
            
            # If there are errors, mark as failure
            if self.validation_results["errors"]:
                self.validation_results["validation_result"] = "failure"
                self._write_error_summary(output_folder)
                success = False
            else:
                self._write_summary(output_folder)
                success = True
            
            # Write log file
            self._write_log(output_folder)
            
            self.logger.info(f"Validation {'passed' if success else 'failed'}")
            return success
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            self._write_error_summary(output_folder, str(e))
            return False
    
    def _validate_file(self, file_name: str, file_path: str):
        """Validate a specific file."""
        file_info = {"valid": False, "error": None}
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                file_info["error"] = "File not found"
                self.validation_results["errors"].append(f"{file_name}: File not found")
            else:
                # Check if file is readable
                if not os.access(file_path, os.R_OK):
                    file_info["error"] = "Permission denied"
                    self.validation_results["errors"].append(f"{file_name}: Permission denied")
                
                # Validate file size
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    file_info["error"] = "File is empty"
                    self.validation_results["errors"].append(f"{file_name}: File is empty")
                else:
                    # File-specific validation
                    if file_name == "metadata.json":
                        self._validate_metadata_json(file_path, file_info)
                    elif file_name == "data.csv":
                        self._validate_data_csv(file_path, file_info)
                    
                    if file_info["valid"]:
                        file_info["size_bytes"] = file_size
                        file_info["checksum"] = self._calculate_checksum(file_path)
        
        except Exception as e:
            file_info["error"] = str(e)
            self.validation_results["errors"].append(f"{file_name}: {str(e)}")
        
        self.validation_results["files_validated"][file_name] = file_info
    
    def _validate_directory(self, dir_name: str, dir_path: str):
        """Validate a specific directory."""
        dir_info = {"valid": False, "error": None}
        
        try:
            # Check if directory exists
            if not os.path.exists(dir_path):
                dir_info["error"] = "Directory not found"
                self.validation_results["errors"].append(f"{dir_name}: Directory not found")
            elif not os.path.isdir(dir_path):
                dir_info["error"] = "Not a directory"
                self.validation_results["errors"].append(f"{dir_name}: Not a directory")
            else:
                # Check if directory has files
                files = [f for f in os.listdir(dir_path) 
                        if os.path.isfile(os.path.join(dir_path, f))]
                
                if not files:
                    dir_info["error"] = "Directory is empty"
                    self.validation_results["errors"].append(f"{dir_name}: Directory is empty")
                else:
                    dir_info["valid"] = True
                    dir_info["file_count"] = len(files)
                    dir_info["files"] = files[:10]  # First 10 files for summary
                    
                    # Check for common asset types
                    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
                    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
                    asset_files = {
                        "images": [],
                        "videos": [],
                        "other": []
                    }
                    
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in image_extensions:
                            asset_files["images"].append(file)
                        elif ext in video_extensions:
                            asset_files["videos"].append(file)
                        else:
                            asset_files["other"].append(file)
                    
                    dir_info["asset_types"] = asset_files
        
        except PermissionError:
            dir_info["error"] = "Permission denied"
            self.validation_results["errors"].append(f"{dir_name}: Permission denied")
        except Exception as e:
            dir_info["error"] = str(e)
            self.validation_results["errors"].append(f"{dir_name}: {str(e)}")
        
        self.validation_results["files_validated"][dir_name] = dir_info
    
    def _validate_metadata_json(self, file_path: str, file_info: Dict):
        """Validate metadata.json structure and content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check required structure
            if "movies" not in metadata:
                file_info["error"] = "Missing 'movies' key"
                self.validation_results["errors"].append("metadata.json: Missing 'movies' key")
                return
            
            movies = metadata["movies"]
            
            if not isinstance(movies, list):
                file_info["error"] = "'movies' should be a list"
                self.validation_results["errors"].append("metadata.json: 'movies' should be a list")
                return
            
            if len(movies) == 0:
                file_info["error"] = "No movies in metadata"
                self.validation_results["errors"].append("metadata.json: No movies in metadata")
                return
            
            # Validate each movie
            required_fields = ["movie_id", "title", "genres"]
            movie_errors = []
            
            for i, movie in enumerate(movies):
                for field in required_fields:
                    if field not in movie:
                        movie_errors.append(f"Movie {i}: Missing '{field}'")
                
                # Validate genres is a list
                if "genres" in movie and not isinstance(movie["genres"], list):
                    movie_errors.append(f"Movie {i}: 'genres' should be a list")
            
            if movie_errors:
                file_info["error"] = f"Movie validation errors: {len(movie_errors)}"
                self.validation_results["errors"].extend(
                    [f"metadata.json: {err}" for err in movie_errors[:5]]
                )
                if len(movie_errors) > 5:
                    self.validation_results["warnings"].append(
                        f"metadata.json: ... and {len(movie_errors) - 5} more movie errors"
                    )
            else:
                file_info["valid"] = True
                file_info["movie_count"] = len(movies)
                
                # Collect genre statistics
                genres = {}
                for movie in movies:
                    for genre in movie.get("genres", []):
                        genres[genre] = genres.get(genre, 0) + 1
                
                file_info["genre_stats"] = {
                    "total_unique": len(genres),
                    "top_genres": dict(sorted(genres.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:10])
                }
                
        except json.JSONDecodeError as e:
            file_info["error"] = f"Invalid JSON: {str(e)}"
            self.validation_results["errors"].append(f"metadata.json: Invalid JSON - {str(e)}")
        except Exception as e:
            file_info["error"] = str(e)
            self.validation_results["errors"].append(f"metadata.json: {str(e)}")
    
    def _validate_data_csv(self, file_path: str, file_info: Dict):
        """Validate data.csv structure and content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Check if file has content
                first_line = f.readline()
                if not first_line:
                    file_info["error"] = "CSV file is empty"
                    self.validation_results["errors"].append("data.csv: File is empty")
                    return
                
                # Check header
                expected_headers = ["user_id", "movie_id", "rating"]
                headers = first_line.strip().split(',')
                
                missing_headers = [h for h in expected_headers if h not in headers]
                if missing_headers:
                    file_info["error"] = f"Missing headers: {missing_headers}"
                    self.validation_results["errors"].append(
                        f"data.csv: Missing required headers: {missing_headers}"
                    )
                    return
                
                # Count rows and validate data
                f.seek(0)
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if len(rows) == 0:
                    file_info["error"] = "No data rows in CSV"
                    self.validation_results["errors"].append("data.csv: No data rows")
                    return
                
                # Validate each row
                row_errors = []
                user_ids = set()
                movie_ids = set()
                ratings = []
                
                for i, row in enumerate(rows, 1):
                    # Check required fields
                    if not row.get("user_id"):
                        row_errors.append(f"Row {i}: Missing user_id")
                    if not row.get("movie_id"):
                        row_errors.append(f"Row {i}: Missing movie_id")
                    
                    # Validate rating if present
                    if "rating" in row and row["rating"]:
                        try:
                            rating = float(row["rating"])
                            if rating < 0 or rating > 5:
                                row_errors.append(f"Row {i}: Rating {rating} out of range (0-5)")
                            ratings.append(rating)
                        except ValueError:
                            row_errors.append(f"Row {i}: Invalid rating format: {row['rating']}")
                    
                    user_ids.add(row.get("user_id", ""))
                    movie_ids.add(row.get("movie_id", ""))
                
                if row_errors:
                    file_info["error"] = f"Data validation errors: {len(row_errors)}"
                    self.validation_results["errors"].extend(
                        [f"data.csv: {err}" for err in row_errors[:5]]
                    )
                    if len(row_errors) > 5:
                        self.validation_results["warnings"].append(
                            f"data.csv: ... and {len(row_errors) - 5} more data errors"
                        )
                else:
                    file_info["valid"] = True
                    file_info["row_count"] = len(rows)
                    file_info["user_count"] = len(user_ids)
                    file_info["movie_count"] = len(movie_ids)
                    
                    if ratings:
                        file_info["rating_stats"] = {
                            "min": min(ratings),
                            "max": max(ratings),
                            "avg": sum(ratings) / len(ratings)
                        }
        
        except csv.Error as e:
            file_info["error"] = f"CSV parsing error: {str(e)}"
            self.validation_results["errors"].append(f"data.csv: CSV parsing error - {str(e)}")
        except Exception as e:
            file_info["error"] = str(e)
            self.validation_results["errors"].append(f"data.csv: {str(e)}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _write_summary(self, output_folder: str):
        """Write summary.json for successful validation."""
        summary_path = os.path.join(output_folder, "summary.json")
        
        # Add processing statistics
        self.validation_results["processing_time"] = {
            "start": self.validation_results["timestamp"],
            "end": datetime.utcnow().isoformat() + "Z"
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        self.logger.info(f"Summary written to: {summary_path}")
    
    def _write_error_summary(self, output_folder: str, fatal_error: str = None):
        """Write error_summary.json for failed validation."""
        error_summary = self.validation_results.copy()
        
        if fatal_error:
            error_summary["fatal_error"] = fatal_error
            error_summary["errors"].insert(0, f"Fatal error: {fatal_error}")
        
        error_path = os.path.join(output_folder, "error_summary.json")
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_summary, f, indent=2, default=str)
        
        self.logger.error(f"Error summary written to: {error_path}")
    
    def _write_log(self, output_folder: str):
        """Write automation.log file."""
        log_path = os.path.join(output_folder, "automation.log")
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"=== Validation Run - {datetime.utcnow().isoformat()}Z ===\n")
            f.write(f"Input folder: {self.validation_results['input_folder']}\n")
            f.write(f"Result: {self.validation_results['validation_result']}\n")
            
            if self.validation_results["errors"]:
                f.write("\nErrors:\n")
                for error in self.validation_results["errors"]:
                    f.write(f"  - {error}\n")
            
            if self.validation_results["warnings"]:
                f.write("\nWarnings:\n")
                for warning in self.validation_results["warnings"]:
                    f.write(f"  - {warning}\n")
            
            f.write("\nFile Validation Summary:\n")
            for file_name, file_info in self.validation_results["files_validated"].items():
                f.write(f"  {file_name}: ")
                if file_info["valid"]:
                    f.write("VALID")
                    if "size_bytes" in file_info:
                        f.write(f" ({file_info['size_bytes']} bytes)")
                    if "movie_count" in file_info:
                        f.write(f" ({file_info['movie_count']} movies)")
                    if "row_count" in file_info:
                        f.write(f" ({file_info['row_count']} rows)")
                else:
                    f.write(f"INVALID - {file_info.get('error', 'Unknown error')}")
                f.write("\n")
            
            f.write("\n" + "="*50 + "\n\n")
        
        self.logger.info(f"Log written to: {log_path}")