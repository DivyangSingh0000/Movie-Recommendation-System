"""
Data loading module for Movie Recommendation System.
Loads processed data into various storage systems.
"""

import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads processed data into storage systems."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'database': {
                'type': 'sqlite',
                'path': 'data/recommendation.db'
            },
            'storage': {
                'processed_data_path': 'data/processed/',
                'models_path': 'models/'
            }
        }
        
        # Create storage directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories for data storage."""
        directories = [
            self.config['storage']['processed_data_path'],
            self.config['storage']['models_path'],
            'data/raw/',
            'logs/'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save_to_disk(self, data: Dict[str, pd.DataFrame], 
                    format_type: str = 'parquet') -> None:
        """
        Save processed data to disk.
        
        Args:
            data: Dictionary of DataFrames to save
            format_type: Format to save in ('parquet', 'csv', 'pickle')
        """
        try:
            logger.info(f"Saving data to disk in {format_type} format")
            save_path = self.config['storage']['processed_data_path']
            
            for name, df in data.items():
                file_path = Path(save_path) / f"{name}.{format_type}"
                
                if format_type == 'parquet':
                    df.to_parquet(file_path, index=False)
                elif format_type == 'csv':
                    df.to_csv(file_path, index=False)
                elif format_type == 'pickle':
                    df.to_pickle(file_path)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
                
                logger.info(f"Saved {name} to {file_path}")
                
        except Exception as e:
            logger.error(f"Error saving data to disk: {str(e)}")
            raise
    
    def load_from_disk(self, data_names: List[str], 
                      format_type: str = 'parquet') -> Dict[str, pd.DataFrame]:
        """
        Load processed data from disk.
        
        Args:
            data_names: List of data names to load
            format_type: Format to load from ('parquet', 'csv', 'pickle')
            
        Returns:
            Dictionary of loaded DataFrames
        """
        try:
            logger.info(f"Loading data from disk in {format_type} format")
            load_path = self.config['storage']['processed_data_path']
            data = {}
            
            for name in data_names:
                file_path = Path(load_path) / f"{name}.{format_type}"
                
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                if format_type == 'parquet':
                    df = pd.read_parquet(file_path)
                elif format_type == 'csv':
                    df = pd.read_csv(file_path)
                elif format_type == 'pickle':
                    df = pd.read_pickle(file_path)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
                
                data[name] = df
                logger.info(f"Loaded {name} from {file_path}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from disk: {str(e)}")
            raise
    
    def save_to_database(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Save data to SQL database.
        
        Args:
            data: Dictionary of DataFrames to save
        """
        try:
            db_config = self.config['database']
            
            if db_config['type'] == 'sqlite':
                conn = sqlite3.connect(db_config['path'])
            else:
                # For other database types, use SQLAlchemy
                engine = create_engine(db_config.get('connection_string'))
                conn = engine.connect()
            
            for table_name, df in data.items():
                # Clean table name
                table_name = table_name.replace(' ', '_').lower()
                
                # Save to database
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logger.info(f"Saved {len(df)} rows to table '{table_name}'")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            raise
    
    def load_from_database(self, table_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load data from SQL database.
        
        Args:
            table_names: List of table names to load
            
        Returns:
            Dictionary of loaded DataFrames
        """
        try:
            db_config = self.config['database']
            data = {}
            
            if db_config['type'] == 'sqlite':
                conn = sqlite3.connect(db_config['path'])
            else:
                engine = create_engine(db_config.get('connection_string'))
                conn = engine.connect()
            
            for table_name in table_names:
                query = f"SELECT * FROM {table_name}"
                df = pd.read_sql_query(query, conn)
                data[table_name] = df
                logger.info(f"Loaded {len(df)} rows from table '{table_name}'")
            
            conn.close()
            return data
            
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            raise
    
    def save_model(self, model: Any, model_name: str, 
                  metadata: Optional[Dict] = None) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Model object to save
            model_name: Name for the model file
            metadata: Additional metadata about the model
        """
        try:
            models_path = Path(self.config['storage']['models_path'])
            model_file = models_path / f"{model_name}.pkl"
            
            # Create model info dictionary
            model_info = {
                'model': model,
                'metadata': metadata or {},
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Save model
            with open(model_file, 'wb') as f:
                pickle.dump(model_info, f)
            
            logger.info(f"Saved model '{model_name}' to {model_file}")
            
            # Save metadata separately as JSON
            if metadata:
                metadata_file = models_path / f"{model_name}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        Load trained model from disk.
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Dictionary containing model and metadata
        """
        try:
            models_path = Path(self.config['storage']['models_path'])
            model_file = models_path / f"{model_name}.pkl"
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            with open(model_file, 'rb') as f:
                model_info = pickle.load(f)
            
            logger.info(f"Loaded model '{model_name}' from {model_file}")
            return model_info
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_recommendations(self, recommendations: Dict[str, List], 
                           user_id: Optional[str] = None) -> None:
        """
        Save recommendations to storage.
        
        Args:
            recommendations: Dictionary of recommendations
            user_id: Optional user ID for personalized recommendations
        """
        try:
            save_path = Path(self.config['storage']['processed_data_path'])
            
            if user_id:
                filename = f"recommendations_{user_id}.json"
            else:
                filename = "recommendations_general.json"
            
            file_path = save_path / filename
            
            # Add metadata
            recommendations_with_meta = {
                'recommendations': recommendations,
                'generated_at': pd.Timestamp.now().isoformat(),
                'user_id': user_id
            }
            
            with open(file_path, 'w') as f:
                json.dump(recommendations_with_meta, f, indent=2)
            
            logger.info(f"Saved recommendations to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
            raise
    
    def load_recommendations(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load recommendations from storage.
        
        Args:
            user_id: Optional user ID for personalized recommendations
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            load_path = Path(self.config['storage']['processed_data_path'])
            
            if user_id:
                filename = f"recommendations_{user_id}.json"
            else:
                filename = "recommendations_general.json"
            
            file_path = load_path / filename
            
            if not file_path.exists():
                logger.warning(f"Recommendations file not found: {file_path}")
                return {}
            
            with open(file_path, 'r') as f:
                recommendations = json.load(f)
            
            logger.info(f"Loaded recommendations from {file_path}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error loading recommendations: {str(e)}")
            raise
