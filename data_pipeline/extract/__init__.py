"""
Data extraction module for Movie Recommendation System.
Extracts data from various sources including CSV, JSON, and databases.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataExtractor:
    """Extracts data from various sources for the recommendation system."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def extract_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Extract data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame containing CSV data
        """
        try:
            logger.info(f"Extracting data from CSV: {file_path}")
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully extracted {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error extracting CSV from {file_path}: {str(e)}")
            raise
    
    def extract_from_json(self, file_path: str) -> Dict:
        """
        Extract data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary containing JSON data
        """
        try:
            logger.info(f"Extracting data from JSON: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully extracted data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error extracting JSON from {file_path}: {str(e)}")
            raise
    
    def extract_metadata(self, metadata_path: str) -> pd.DataFrame:
        """
        Extract movie metadata from JSON file.
        
        Args:
            metadata_path: Path to metadata.json
            
        Returns:
            DataFrame containing movie metadata
        """
        try:
            metadata = self.extract_from_json(metadata_path)
            movies = metadata.get('movies', [])
            
            if not movies:
                raise ValueError("No movies found in metadata")
            
            # Convert to DataFrame
            df_metadata = pd.DataFrame(movies)
            
            # Flatten nested structures if needed
            if 'genres' in df_metadata.columns:
                df_metadata['genres'] = df_metadata['genres'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else x
                )
            
            logger.info(f"Extracted metadata for {len(df_metadata)} movies")
            return df_metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise
    
    def extract_user_interactions(self, interactions_path: str) -> pd.DataFrame:
        """
        Extract user-movie interactions from CSV file.
        
        Args:
            interactions_path: Path to data.csv
            
        Returns:
            DataFrame containing user interactions
        """
        try:
            df_interactions = self.extract_from_csv(interactions_path)
            
            # Ensure required columns exist
            required_columns = ['user_id', 'movie_id', 'rating']
            missing_columns = [col for col in required_columns 
                             if col not in df_interactions.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Extracted {len(df_interactions)} user interactions")
            return df_interactions
            
        except Exception as e:
            logger.error(f"Error extracting user interactions: {str(e)}")
            raise
    
    def extract_assets_info(self, assets_dir: str) -> pd.DataFrame:
        """
        Extract information about assets in the assets directory.
        
        Args:
            assets_dir: Path to assets directory
            
        Returns:
            DataFrame containing asset information
        """
        try:
            assets_info = []
            
            for root, dirs, files in os.walk(assets_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, assets_dir)
                    
                    # Extract movie_id from filename if possible
                    movie_id = self._extract_movie_id_from_filename(file)
                    
                    asset_info = {
                        'file_name': file,
                        'file_path': rel_path,
                        'full_path': file_path,
                        'file_extension': os.path.splitext(file)[1].lower(),
                        'file_size': os.path.getsize(file_path),
                        'movie_id': movie_id,
                        'directory': root
                    }
                    assets_info.append(asset_info)
            
            df_assets = pd.DataFrame(assets_info)
            logger.info(f"Extracted info for {len(df_assets)} assets")
            return df_assets
            
        except Exception as e:
            logger.error(f"Error extracting assets info: {str(e)}")
            raise
    
    def _extract_movie_id_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract movie_id from filename.
        
        Args:
            filename: Filename to extract movie_id from
            
        Returns:
            Extracted movie_id or None
        """
        # Common patterns: movie_id.jpg, tt1234567.jpg, movie-poster-tt1234567.jpg
        import re
        
        # Pattern for IMDb IDs (tt followed by 7-8 digits)
        imdb_pattern = r'(tt\d{7,8})'
        match = re.search(imdb_pattern, filename)
        
        if match:
            return match.group(1)
        
        # Pattern for numeric IDs
        numeric_pattern = r'(\d{3,})'
        match = re.search(numeric_pattern, filename.split('.')[0])
        
        if match:
            return match.group(1)
        
        return None
    
    def extract_all(self, input_folder: str) -> Dict[str, pd.DataFrame]:
        """
        Extract all data from input folder.
        
        Args:
            input_folder: Path to input folder
            
        Returns:
            Dictionary of DataFrames for each data type
        """
        try:
            logger.info(f"Extracting all data from: {input_folder}")
            
            # Check if folder exists
            if not os.path.exists(input_folder):
                raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
            data = {}
            
            # Extract metadata
            metadata_path = os.path.join(input_folder, 'metadata.json')
            if os.path.exists(metadata_path):
                data['metadata'] = self.extract_metadata(metadata_path)
            
            # Extract user interactions
            interactions_path = os.path.join(input_folder, 'data.csv')
            if os.path.exists(interactions_path):
                data['interactions'] = self.extract_user_interactions(interactions_path)
            
            # Extract assets info
            assets_dir = os.path.join(input_folder, 'assets')
            if os.path.exists(assets_dir):
                data['assets'] = self.extract_assets_info(assets_dir)
            
            logger.info(f"Successfully extracted all data")
            return data
            
        except Exception as e:
            logger.error(f"Error in extract_all: {str(e)}")
            raise
