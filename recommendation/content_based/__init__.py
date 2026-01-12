# movie-recommendation-system/recommendation/content_based/__init__.py
"""
Content-based recommendation algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import re

logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    """Content-based recommendation system."""
    
    def __init__(self, feature_weights: Optional[Dict] = None):
        self.feature_weights = feature_weights or {
            'genres': 0.4,
            'description': 0.3,
            'directors': 0.1,
            'actors': 0.1,
            'year': 0.05,
            'rating': 0.05
        }
        self.movie_features = None
        self.similarity_matrix = None
        self.movie_ids = None
        self.vectorizers = {}
        self.scalers = {}
        
    def prepare_features(self, df_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for content-based recommendation.
        
        Args:
            df_metadata: Movie metadata DataFrame
            
        Returns:
            DataFrame with prepared features
        """
        try:
            logger.info("Preparing features for content-based recommendation")
            df = df_metadata.copy()
            
            # Ensure movie_id is present
            if 'movie_id' not in df.columns:
                raise ValueError("DataFrame must contain 'movie_id' column")
            
            self.movie_ids = df['movie_id'].tolist()
            
            # Prepare genre features
            if 'genres' in df.columns:
                df = self._prepare_genre_features(df)
            
            # Prepare text features
            text_columns = ['description', 'title']
            for col in text_columns:
                if col in df.columns:
                    df = self._prepare_text_features(df, col)
            
            # Prepare categorical features
            categorical_cols = ['directors', 'actors', 'language', 'country']
            for col in categorical_cols:
                if col in df.columns:
                    df = self._prepare_categorical_features(df, col)
            
            # Prepare numeric features
            numeric_cols = ['release_year', 'duration_minutes', 'imdb_rating']
            numeric_cols = [col for col in numeric_cols if col in df.columns]
            if numeric_cols:
                df = self._prepare_numeric_features(df, numeric_cols)
            
            # Combine all features into a single feature matrix
            feature_columns = [col for col in df.columns 
                             if col.startswith('feat_') or col.startswith('tfidf_')]
            
            if not feature_columns:
                raise ValueError("No features were created from the data")
            
            self.movie_features = df[['movie_id'] + feature_columns]
            logger.info(f"Prepared {len(feature_columns)} features for {len(df)} movies")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def fit(self, df_metadata: pd.DataFrame) -> 'ContentBasedRecommender':
        """
        Fit content-based recommendation model.
        
        Args:
            df_metadata: Movie metadata DataFrame
            
        Returns:
            Self
        """
        try:
            # Prepare features
            df = self.prepare_features(df_metadata)
            
            # Extract feature matrix (excluding movie_id)
            feature_columns = [col for col in df.columns 
                             if col.startswith('feat_') or col.startswith('tfidf_')]
            feature_matrix = df[feature_columns].values
            
            # Apply feature weights if specified
            if self.feature_weights:
                weighted_matrix = self._apply_feature_weights(feature_matrix, feature_columns)
                feature_matrix = weighted_matrix
            
            # Compute similarity matrix
            logger.info("Computing content similarity matrix")
            self.similarity_matrix = cosine_similarity(feature_matrix)
            
            # Convert to DataFrame for easier indexing
            self.similarity_matrix = pd.DataFrame(
                self.similarity_matrix,
                index=self.movie_ids,
                columns=self.movie_ids
            )
            
            logger.info("Content-based model fitted")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting content-based model: {str(e)}")
            raise
    
    def recommend_similar(self, movie_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Recommend movies similar to a given movie.
        
        Args:
            movie_id: Movie ID to find similar movies for
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        try:
            if movie_id not in self.movie_ids:
                return []
            
            # Get similarity scores for this movie
            similarities = self.similarity_matrix[movie_id].sort_values(ascending=False)
            
            # Exclude the movie itself
            similar_movies = [(mid, score) for mid, score in similarities.items() 
                             if mid != movie_id]
            
            return similar_movies[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in content-based recommendation: {str(e)}")
            return []
    
    def recommend_for_user(self, user_ratings: Dict[str, float], 
                          n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Recommend movies for a user based on their ratings.
        
        Args:
            user_ratings: Dictionary of {movie_id: rating}
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        try:
            if not user_ratings:
                return []
            
            # Calculate weighted average of similarity vectors
            weighted_similarities = pd.Series(0, index=self.movie_ids)
            
            for movie_id, rating in user_ratings.items():
                if movie_id in self.movie_ids:
                    # Weight by rating (higher rating = more influence)
                    similarity_vector = self.similarity_matrix[movie_id] * rating
                    weighted_similarities += similarity_vector
            
            # Normalize by number of rated movies
            weighted_similarities /= len(user_ratings)
            
            # Exclude already rated movies
            for movie_id in user_ratings.keys():
                if movie_id in weighted_similarities.index:
                    weighted_similarities[movie_id] = -1  # Mark as rated
            
            # Get top recommendations
            recommendations = weighted_similarities.sort_values(ascending=False)
            top_recommendations = [(mid, score) for mid, score in recommendations.items() 
                                  if score >= 0][:n_recommendations]
            
            return top_recommendations
            
        except Exception as e:
            logger.error(f"Error recommending for user: {str(e)}")
            return []
    
    def _prepare_genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare genre features using multi-label binarization."""
        try:
            # Convert genres to list if they're strings
            if df['genres'].dtype == 'object':
                df['genres_list'] = df['genres'].apply(self._parse_genres)
            else:
                df['genres_list'] = df['genres']
            
            # Create multi-label binarizer
            mlb = MultiLabelBinarizer()
            genre_matrix = mlb.fit_transform(df['genres_list'])
            
            # Create feature columns
            genre_columns = [f'feat_genre_{genre}' for genre in mlb.classes_]
            genre_df = pd.DataFrame(genre_matrix, columns=genre_columns, index=df.index)
            
            # Concatenate with original dataframe
            df = pd.concat([df, genre_df], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing genre features: {str(e)}")
            return df
    
    def _prepare_text_features(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Prepare text features using TF-IDF."""
        try:
            # Clean text
            df[f'{column}_clean'] = df[column].fillna('').astype(str).apply(self._clean_text)
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(df[f'{column}_clean'])
            
            # Create feature columns
            feature_names = vectorizer.get_feature_names_out()
            tfidf_columns = [f'tfidf_{column}_{name}' for name in feature_names]
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=tfidf_columns,
                index=df.index
            )
            
            # Store vectorizer for later use
            self.vectorizers[column] = vectorizer
            
            # Concatenate with original dataframe
            df = pd.concat([df, tfidf_df], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing text features for {column}: {str(e)}")
            return df
    
    def _prepare_categorical_features(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Prepare categorical features."""
        try:
            # Convert to string and handle lists
            if df[column].dtype == 'object':
                df[f'{column}_str'] = df[column].astype(str)
                
                # For list-like strings, extract first element
                df[f'{column}_str'] = df[f'{column}_str'].apply(
                    lambda x: x.split(',')[0].strip('[]\'" ') if ',' in x else x
                )
            else:
                df[f'{column}_str'] = df[column].astype(str)
            
            # One-hot encode if not too many unique values
            unique_values = df[f'{column}_str'].nunique()
            if unique_values <= 50:
                dummies = pd.get_dummies(df[f'{column}_str'], prefix=f'feat_{column}')
                df = pd.concat([df, dummies], axis=1)
            else:
                # Use label encoding for high cardinality
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[f'feat_{column}_encoded'] = le.fit_transform(df[f'{column}_str'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing categorical features for {column}: {str(e)}")
            return df
    
    def _prepare_numeric_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Prepare and scale numeric features."""
        try:
            for col in columns:
                # Fill missing values
                df[col] = df[col].fillna(df[col].median())
                
                # Scale the feature
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(df[[col]])
                df[f'feat_{col}_scaled'] = scaled_values
                
                # Store scaler for later use
                self.scalers[col] = scaler
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing numeric features: {str(e)}")
            return df
    
    def _apply_feature_weights(self, feature_matrix: np.ndarray, 
                              feature_columns: List[str]) -> np.ndarray:
        """Apply weights to features based on configuration."""
        weighted_matrix = feature_matrix.copy()
        
        # Group features by type
        feature_groups = {}
        for i, col in enumerate(feature_columns):
            for feat_type in self.feature_weights.keys():
                if col.startswith(f'feat_{feat_type}') or col.startswith(f'tfidf_{feat_type}'):
                    if feat_type not in feature_groups:
                        feature_groups[feat_type] = []
                    feature_groups[feat_type].append(i)
                    break
        
        # Apply weights
        for feat_type, indices in feature_groups.items():
            if feat_type in self.feature_weights:
                weight = self.feature_weights[feat_type]
                weighted_matrix[:, indices] *= weight
        
        return weighted_matrix
    
    def _parse_genres(self, genres_str: Any) -> List[str]:
        """Parse genres string to list."""
        if isinstance(genres_str, list):
            return [str(g).strip() for g in genres_str]
        elif isinstance(genres_str, str):
            # Handle different formats
            genres_str = genres_str.strip("[]")
            genres = [g.strip() for g in genres_str.split(',')]
            return [g.strip('\'"') for g in genres]
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