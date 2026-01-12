"""
Data transformation module for Movie Recommendation System.
Transforms raw data into formats suitable for analysis and modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re

logger = logging.getLogger(__name__)

class DataTransformer:
    """Transforms data for the recommendation system."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.vectorizers = {}
    
    def clean_metadata(self, df_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess movie metadata.
        
        Args:
            df_metadata: Raw metadata DataFrame
            
        Returns:
            Cleaned metadata DataFrame
        """
        try:
            logger.info("Cleaning movie metadata")
            df = df_metadata.copy()
            
            # Ensure movie_id is string and remove duplicates
            if 'movie_id' in df.columns:
                df['movie_id'] = df['movie_id'].astype(str)
                df = df.drop_duplicates(subset=['movie_id'])
            
            # Clean title
            if 'title' in df.columns:
                df['title'] = df['title'].astype(str).str.strip()
                df['title_clean'] = df['title'].str.lower()
            
            # Process genres
            if 'genres' in df.columns:
                # Convert string representation of list to actual list
                if df['genres'].dtype == 'object':
                    df['genres'] = df['genres'].apply(
                        lambda x: self._parse_genres(x)
                    )
                df['genre_count'] = df['genres'].apply(len)
                df['all_genres'] = df['genres'].apply(
                    lambda x: ' '.join(x) if isinstance(x, list) else ''
                )
            
            # Clean description/text fields
            text_columns = ['description', 'plot', 'overview']
            for col in text_columns:
                if col in df.columns:
                    df[f'{col}_clean'] = df[col].astype(str).apply(self._clean_text)
            
            # Process numeric fields
            numeric_columns = ['release_year', 'duration_minutes', 'imdb_rating']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            logger.info(f"Cleaned metadata for {len(df)} movies")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning metadata: {str(e)}")
            raise
    
    def clean_interactions(self, df_interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess user interactions.
        
        Args:
            df_interactions: Raw interactions DataFrame
            
        Returns:
            Cleaned interactions DataFrame
        """
        try:
            logger.info("Cleaning user interactions")
            df = df_interactions.copy()
            
            # Convert IDs to strings
            if 'user_id' in df.columns:
                df['user_id'] = df['user_id'].astype(str)
            
            if 'movie_id' in df.columns:
                df['movie_id'] = df['movie_id'].astype(str)
            
            # Clean rating column
            if 'rating' in df.columns:
                df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
                # Remove ratings outside valid range (0-5 or 0-10)
                df['rating'] = df['rating'].clip(lower=0, upper=5)
            
            # Parse timestamp if exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['timestamp'] = df['timestamp'].fillna(pd.Timestamp.now())
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Remove interactions with missing essential data
            essential_columns = ['user_id', 'movie_id']
            df = df.dropna(subset=essential_columns)
            
            logger.info(f"Cleaned {len(df)} interactions")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning interactions: {str(e)}")
            raise
    
    def create_user_features(self, df_interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Create user features from interactions.
        
        Args:
            df_interactions: Cleaned interactions DataFrame
            
        Returns:
            DataFrame with user features
        """
        try:
            logger.info("Creating user features")
            
            user_features = df_interactions.groupby('user_id').agg({
                'rating': ['count', 'mean', 'std', 'min', 'max'],
                'movie_id': 'nunique'
            }).reset_index()
            
            # Flatten column names
            user_features.columns = [
                'user_id',
                'interaction_count',
                'avg_rating',
                'rating_std',
                'min_rating',
                'max_rating',
                'unique_movies_rated'
            ]
            
            # Calculate user activity level
            # Calculate user activity level; fall back when there are too few unique values
            try:
                if user_features['interaction_count'].nunique() >= 4:
                    user_features['activity_level'] = pd.qcut(
                        user_features['interaction_count'],
                        q=4,
                        labels=['low', 'medium', 'high', 'very_high']
                    )
                else:
                    # Not enough distinct values to quantile; assign 'low' for now
                    user_features['activity_level'] = 'low'
            except Exception:
                user_features['activity_level'] = 'low'
            
            # Calculate rating tendency
            user_features['rating_tendency'] = user_features['avg_rating'].apply(
                lambda x: 'harsh' if x < 2.5 else 'average' if x < 3.5 else 'generous'
            )
            
            logger.info(f"Created features for {len(user_features)} users")
            return user_features
            
        except Exception as e:
            logger.error(f"Error creating user features: {str(e)}")
            raise
    
    def create_movie_features(self, df_metadata: pd.DataFrame, 
                            df_interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Create movie features from metadata and interactions.
        
        Args:
            df_metadata: Cleaned metadata DataFrame
            df_interactions: Cleaned interactions DataFrame
            
        Returns:
            DataFrame with movie features
        """
        try:
            logger.info("Creating movie features")
            
            # Start with metadata features
            movie_features = df_metadata.copy()
            
            # Add interaction statistics if available
            if df_interactions is not None and not df_interactions.empty:
                interaction_stats = df_interactions.groupby('movie_id').agg({
                    'rating': ['count', 'mean', 'std'],
                    'user_id': 'nunique'
                }).reset_index()
                
                interaction_stats.columns = [
                    'movie_id',
                    'rating_count',
                    'avg_rating',
                    'rating_std',
                    'unique_users'
                ]
                
                movie_features = pd.merge(
                    movie_features,
                    interaction_stats,
                    on='movie_id',
                    how='left'
                )
                
                # Fill missing values
                movie_features['rating_count'] = movie_features['rating_count'].fillna(0)
                movie_features['unique_users'] = movie_features['unique_users'].fillna(0)
                
                # Calculate popularity score
                movie_features['popularity_score'] = (
                    movie_features['rating_count'] * 0.7 +
                    movie_features['unique_users'] * 0.3
                )
            
            # Encode categorical features
            categorical_cols = ['genres', 'language', 'country']
            for col in categorical_cols:
                if col in movie_features.columns:
                    movie_features = self._encode_categorical(movie_features, col)
            
            # Create TF-IDF features from text
            text_cols = ['description_clean', 'all_genres']
            for col in text_cols:
                if col in movie_features.columns:
                    movie_features = self._create_text_features(movie_features, col)
            
            # Normalize numeric features
            numeric_cols = ['release_year', 'duration_minutes', 'imdb_rating']
            numeric_cols = [col for col in numeric_cols if col in movie_features.columns]
            
            if numeric_cols:
                movie_features = self._normalize_features(movie_features, numeric_cols)
            
            logger.info(f"Created features for {len(movie_features)} movies")
            return movie_features
            
        except Exception as e:
            logger.error(f"Error creating movie features: {str(e)}")
            raise
    
    def create_interaction_matrix(self, df_interactions: pd.DataFrame, 
                                df_users: pd.DataFrame,
                                df_movies: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction matrix for collaborative filtering.
        
        Args:
            df_interactions: Cleaned interactions DataFrame
            df_users: User features DataFrame
            df_movies: Movie features DataFrame
            
        Returns:
            Interaction matrix DataFrame
        """
        try:
            logger.info("Creating interaction matrix")
            
            # Create user-item matrix
            interaction_matrix = df_interactions.pivot_table(
                index='user_id',
                columns='movie_id',
                values='rating',
                aggfunc='mean'
            )
            
            # Fill missing values with 0 (or appropriate default)
            interaction_matrix = interaction_matrix.fillna(0)
            
            # Ensure all users and movies are included
            all_users = set(df_users['user_id'])
            all_movies = set(df_movies['movie_id'])
            
            # Add missing users
            missing_users = all_users - set(interaction_matrix.index)
            for user in missing_users:
                interaction_matrix.loc[user] = 0
            
            # Add missing movies
            missing_movies = all_movies - set(interaction_matrix.columns)
            for movie in missing_movies:
                interaction_matrix[movie] = 0
            
            logger.info(f"Created interaction matrix: {interaction_matrix.shape}")
            return interaction_matrix
            
        except Exception as e:
            logger.error(f"Error creating interaction matrix: {str(e)}")
            raise
    
    def _parse_genres(self, genres_str: Any) -> List[str]:
        """Parse genres string to list."""
        if isinstance(genres_str, list):
            return [str(g).strip() for g in genres_str]
        elif isinstance(genres_str, str):
            # Handle different formats: "Action, Adventure" or "[Action, Adventure]"
            genres_str = genres_str.strip("[]")
            genres = [g.strip() for g in genres_str.split(',')]
            return genres
        return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if not df[col].mode().empty:
                mode_val = df[col].mode()[0]
                # If mode is a list or non-scalar, convert to string
                if isinstance(mode_val, (list, tuple)):
                    mode_val = ' '.join(map(str, mode_val))
            else:
                mode_val = 'Unknown'
            df[col] = df[col].fillna(mode_val)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Encode categorical column using LabelEncoder."""
        if column not in self.label_encoders:
            self.label_encoders[column] = LabelEncoder()
        
        df[f'{column}_encoded'] = self.label_encoders[column].fit_transform(
            df[column].fillna('Unknown').astype(str)
        )
        return df
    
    def _create_text_features(self, df: pd.DataFrame, column: str, 
                            max_features: int = 100) -> pd.DataFrame:
        """Create TF-IDF features from text column."""
        vectorizer_key = f'{column}_tfidf'
        
        if vectorizer_key not in self.vectorizers:
            self.vectorizers[vectorizer_key] = TfidfVectorizer(
                max_features=max_features,
                stop_words='english'
            )
        
        vectorizer = self.vectorizers[vectorizer_key]
        tfidf_matrix = vectorizer.fit_transform(df[column].fillna(''))
        
        # Convert to DataFrame
        feature_names = vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{column}_{name}' for name in feature_names]
        )
        
        # Reset index to match original df
        tfidf_df.index = df.index
        
        # Concatenate with original df
        df = pd.concat([df, tfidf_df], axis=1)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize numeric features using MinMaxScaler."""
        for col in columns:
            scaler_key = f'{col}_scaler'
            
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = MinMaxScaler()
            
            scaler = self.scalers[scaler_key]
            df[f'{col}_normalized'] = scaler.fit_transform(df[[col]])
        
        return df
