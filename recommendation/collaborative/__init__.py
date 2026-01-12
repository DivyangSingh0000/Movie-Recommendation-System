"""
Collaborative filtering recommendation algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import pickle

logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    """Base class for collaborative filtering algorithms."""
    
    def __init__(self):
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.interaction_matrix = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, interaction_matrix: pd.DataFrame) -> 'CollaborativeFiltering':
        """
        Fit the collaborative filtering model.
        
        Args:
            interaction_matrix: User-item interaction matrix
            
        Returns:
            Self
        """
        self.interaction_matrix = interaction_matrix
        self.user_ids = interaction_matrix.index.tolist()
        self.item_ids = interaction_matrix.columns.tolist()
        return self
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        raise NotImplementedError
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        raise NotImplementedError

class UserBasedCF(CollaborativeFiltering):
    """User-based collaborative filtering."""
    
    def __init__(self, similarity_metric: str = 'cosine', 
                 min_similar_users: int = 5):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.min_similar_users = min_similar_users
        self.user_similarity_matrix = None
    
    def fit(self, interaction_matrix: pd.DataFrame) -> 'UserBasedCF':
        """
        Fit user-based collaborative filtering model.
        
        Args:
            interaction_matrix: User-item interaction matrix
            
        Returns:
            Self
        """
        super().fit(interaction_matrix)
        
        # Compute user similarity matrix
        logger.info("Computing user similarity matrix")
        if self.similarity_metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(
                self.interaction_matrix.values
            )
        elif self.similarity_metric == 'pearson':
            # Use Pearson correlation
            self.user_similarity_matrix = np.corrcoef(
                self.interaction_matrix.values.T
            )
        
        # Convert to DataFrame for easier indexing
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_ids,
            columns=self.user_ids
        )
        
        logger.info("User-based CF model fitted")
        return self
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating using user-based collaborative filtering.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        try:
            if user_id not in self.user_ids or item_id not in self.item_ids:
                return 0.0
            
            # Get user's mean rating
            user_mean = self.interaction_matrix.loc[user_id].mean()
            
            # Find similar users who rated this item
            similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)
            
            numerator = 0
            denominator = 0
            count = 0
            
            for other_user, similarity in similar_users.items():
                if other_user == user_id:
                    continue
                
                rating = self.interaction_matrix.loc[other_user, item_id]
                if rating > 0:  # User has rated this item
                    other_user_mean = self.interaction_matrix.loc[other_user].mean()
                    numerator += similarity * (rating - other_user_mean)
                    denominator += abs(similarity)
                    count += 1
                
                if count >= self.min_similar_users:
                    break
            
            if denominator == 0:
                return user_mean
            
            predicted = user_mean + (numerator / denominator)
            return max(0, min(5, predicted))  # Clip to rating range
            
        except Exception as e:
            logger.error(f"Error in user-based prediction: {str(e)}")
            return 0.0
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations using user-based CF.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        try:
            if user_id not in self.user_ids:
                return []
            
            # Get user's rated items
            user_ratings = self.interaction_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            
            # Predict ratings for unrated items
            predictions = []
            for item_id in self.item_ids:
                if item_id not in rated_items:
                    pred_rating = self.predict(user_id, item_id)
                    if pred_rating > 0:
                        predictions.append((item_id, pred_rating))
            
            # Sort by predicted rating and return top N
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in user-based recommendation: {str(e)}")
            return []

class ItemBasedCF(CollaborativeFiltering):
    """Item-based collaborative filtering."""
    
    def __init__(self, similarity_metric: str = 'cosine', 
                 min_similar_items: int = 5):
        super().__init__()
        self.similarity_metric = similarity_metric
        self.min_similar_items = min_similar_items
        self.item_similarity_matrix = None
    
    def fit(self, interaction_matrix: pd.DataFrame) -> 'ItemBasedCF':
        """
        Fit item-based collaborative filtering model.
        
        Args:
            interaction_matrix: User-item interaction matrix
            
        Returns:
            Self
        """
        super().fit(interaction_matrix)
        
        # Compute item similarity matrix
        logger.info("Computing item similarity matrix")
        if self.similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(
                self.interaction_matrix.values.T
            )
        
        # Convert to DataFrame for easier indexing
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.item_ids,
            columns=self.item_ids
        )
        
        logger.info("Item-based CF model fitted")
        return self
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating using item-based collaborative filtering.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        try:
            if user_id not in self.user_ids or item_id not in self.item_ids:
                return 0.0
            
            # Get user's ratings
            user_ratings = self.interaction_matrix.loc[user_id]
            
            # Find similar items that the user has rated
            similar_items = self.item_similarity_matrix[item_id].sort_values(ascending=False)
            
            numerator = 0
            denominator = 0
            count = 0
            
            for other_item, similarity in similar_items.items():
                if other_item == item_id:
                    continue
                
                rating = user_ratings[other_item]
                if rating > 0:  # User has rated this item
                    numerator += similarity * rating
                    denominator += abs(similarity)
                    count += 1
                
                if count >= self.min_similar_items:
                    break
            
            if denominator == 0:
                return 0.0
            
            predicted = numerator / denominator
            return max(0, min(5, predicted))  # Clip to rating range
            
        except Exception as e:
            logger.error(f"Error in item-based prediction: {str(e)}")
            return 0.0
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations using item-based CF.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        try:
            if user_id not in self.user_ids:
                return []
            
            # Get user's rated items
            user_ratings = self.interaction_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            
            # Predict ratings for unrated items
            predictions = []
            for item_id in self.item_ids:
                if item_id not in rated_items:
                    pred_rating = self.predict(user_id, item_id)
                    if pred_rating > 0:
                        predictions.append((item_id, pred_rating))
            
            # Sort by predicted rating and return top N
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in item-based recommendation: {str(e)}")
            return []

class MatrixFactorization(CollaborativeFiltering):
    """Matrix Factorization using SVD."""
    
    def __init__(self, n_factors: int = 50, 
                 regularization: float = 0.02,
                 learning_rate: float = 0.01,
                 n_epochs: int = 20):
        super().__init__()
        self.n_factors = n_factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.global_bias = None
        self.user_biases = None
        self.item_biases = None
    
    def fit(self, interaction_matrix: pd.DataFrame) -> 'MatrixFactorization':
        """
        Fit matrix factorization model using stochastic gradient descent.
        
        Args:
            interaction_matrix: User-item interaction matrix
            
        Returns:
            Self
        """
        super().fit(interaction_matrix)
        
        logger.info(f"Training matrix factorization with {self.n_factors} factors")
        
        # Initialize factors and biases
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_bias = np.mean(self.interaction_matrix.values[self.interaction_matrix.values > 0])
        
        # Create user and item mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}
        
        # Get non-zero ratings for training
        rows, cols = np.where(self.interaction_matrix.values > 0)
        ratings = self.interaction_matrix.values[rows, cols]
        
        # Stochastic Gradient Descent
        for epoch in range(self.n_epochs):
            total_error = 0
            indices = np.arange(len(rows))
            np.random.shuffle(indices)
            
            for idx in indices:
                u = rows[idx]
                i = cols[idx]
                r = ratings[idx]
                
                # Current prediction
                prediction = (
                    self.global_bias + 
                    self.user_biases[u] + 
                    self.item_biases[i] +
                    np.dot(self.user_factors[u], self.item_factors[i])
                )
                
                # Error
                error = r - prediction
                total_error += error ** 2
                
                # Update biases
                self.user_biases[u] += self.learning_rate * (
                    error - self.regularization * self.user_biases[u]
                )
                self.item_biases[i] += self.learning_rate * (
                    error - self.regularization * self.item_biases[i]
                )
                
                # Update factors
                user_factor_u = self.user_factors[u].copy()
                item_factor_i = self.item_factors[i].copy()
                
                self.user_factors[u] += self.learning_rate * (
                    error * item_factor_i - self.regularization * self.user_factors[u]
                )
                self.item_factors[i] += self.learning_rate * (
                    error * user_factor_u - self.regularization * self.item_factors[i]
                )
            
            # Print progress
            rmse = np.sqrt(total_error / len(rows))
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")
        
        logger.info("Matrix factorization model fitted")
        return self
    
    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predict rating using matrix factorization.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        try:
            if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
                return 0.0
            
            u = self.user_to_idx[user_id]
            i = self.item_to_idx[item_id]
            
            prediction = (
                self.global_bias + 
                self.user_biases[u] + 
                self.item_biases[i] +
                np.dot(self.user_factors[u], self.item_factors[i])
            )
            
            return max(0, min(5, prediction))  # Clip to rating range
            
        except Exception as e:
            logger.error(f"Error in MF prediction: {str(e)}")
            return 0.0
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations using matrix factorization.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        try:
            if user_id not in self.user_to_idx:
                return []
            
            u = self.user_to_idx[user_id]
            
            # Get user's rated items
            user_ratings = self.interaction_matrix.loc[user_id]
            rated_items = user_ratings[user_ratings > 0].index.tolist()
            
            # Predict ratings for all items
            predictions = []
            for item_id in self.item_ids:
                if item_id not in rated_items:
                    pred_rating = self.predict(user_id, item_id)
                    if pred_rating > 0:
                        predictions.append((item_id, pred_rating))
            
            # Sort by predicted rating and return top N
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in MF recommendation: {str(e)}")
            return []
    
    def get_similar_items(self, item_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """
        Get similar items based on latent factors.
        
        Args:
            item_id: Item ID
            n_similar: Number of similar items to return
            
        Returns:
            List of (similar_item_id, similarity_score) tuples
        """
        try:
            if item_id not in self.item_to_idx:
                return []
            
            i = self.item_to_idx[item_id]
            item_factor = self.item_factors[i]
            
            # Compute similarities with all other items
            similarities = []
            for other_item_id, j in self.item_to_idx.items():
                if other_item_id != item_id:
                    other_factor = self.item_factors[j]
                    similarity = np.dot(item_factor, other_factor) / (
                        np.linalg.norm(item_factor) * np.linalg.norm(other_factor)
                    )
                    similarities.append((other_item_id, similarity))
            
            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:n_similar]
            
        except Exception as e:
            logger.error(f"Error getting similar items: {str(e)}")
            return []
