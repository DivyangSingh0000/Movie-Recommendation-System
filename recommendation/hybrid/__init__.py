"""
Hybrid recommendation algorithms combining multiple approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches."""
    
    def __init__(self, recommenders: Dict[str, Any], 
                 weights: Optional[Dict[str, float]] = None,
                 strategy: str = 'weighted'):
        """
        Initialize hybrid recommender.
        
        Args:
            recommenders: Dictionary of recommender instances
            weights: Dictionary of weights for each recommender
            strategy: Combination strategy ('weighted', 'switching', 'cascade')
        """
        self.recommenders = recommenders
        self.weights = weights or {name: 1.0/len(recommenders) 
                                  for name in recommenders.keys()}
        self.strategy = strategy
        self.user_profiles = {}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def fit(self, data: Dict[str, pd.DataFrame]) -> 'HybridRecommender':
        """
        Fit all component recommenders.
        
        Args:
            data: Dictionary containing data for fitting
            
        Returns:
            Self
        """
        try:
            logger.info("Fitting hybrid recommender components")
            
            for name, recommender in self.recommenders.items():
                logger.info(f"Fitting {name} recommender")
                
                if hasattr(recommender, 'fit'):
                    # Determine what data this recommender needs
                    if 'collaborative' in name.lower():
                        if 'interaction_matrix' in data:
                            recommender.fit(data['interaction_matrix'])
                    elif 'content' in name.lower():
                        if 'metadata' in data:
                            recommender.fit(data['metadata'])
                    elif 'matrix' in name.lower() or 'factorization' in name.lower():
                        if 'interaction_matrix' in data:
                            recommender.fit(data['interaction_matrix'])
            
            logger.info("Hybrid recommender fitted")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting hybrid recommender: {str(e)}")
            raise
    
    def recommend(self, user_id: str, n_recommendations: int = 10,
                 context: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Generate hybrid recommendations.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            context: Additional context information
            
        Returns:
            List of (movie_id, score) tuples
        """
        try:
            if self.strategy == 'weighted':
                return self._weighted_recommendation(user_id, n_recommendations, context)
            elif self.strategy == 'switching':
                return self._switching_recommendation(user_id, n_recommendations, context)
            elif self.strategy == 'cascade':
                return self._cascade_recommendation(user_id, n_recommendations, context)
            elif self.strategy == 'mixed':
                return self._mixed_recommendation(user_id, n_recommendations, context)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
                
        except Exception as e:
            logger.error(f"Error in hybrid recommendation: {str(e)}")
            return []
    
    def _weighted_recommendation(self, user_id: str, n_recommendations: int,
                               context: Optional[Dict]) -> List[Tuple[str, float]]:
        """Weighted combination of recommendations."""
        all_recommendations = {}
        
        # Get recommendations from each recommender
        for name, recommender in self.recommenders.items():
            try:
                if hasattr(recommender, 'recommend'):
                    recommendations = recommender.recommend(user_id, n_recommendations * 2)
                    
                    # Apply weight to scores
                    weight = self.weights.get(name, 0)
                    for movie_id, score in recommendations:
                        if movie_id not in all_recommendations:
                            all_recommendations[movie_id] = 0
                        all_recommendations[movie_id] += score * weight
            except Exception as e:
                logger.warning(f"Recommender {name} failed: {str(e)}")
        
        # Sort by weighted score
        sorted_recommendations = sorted(all_recommendations.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
        
        return sorted_recommendations[:n_recommendations]
    
    def _switching_recommendation(self, user_id: str, n_recommendations: int,
                                context: Optional[Dict]) -> List[Tuple[str, float]]:
        """Switch between recommenders based on context or user profile."""
        # Determine which recommender to use
        recommender_name = self._select_recommender(user_id, context)
        
        if recommender_name in self.recommenders:
            recommender = self.recommenders[recommender_name]
            return recommender.recommend(user_id, n_recommendations)
        
        # Fallback to weighted combination
        return self._weighted_recommendation(user_id, n_recommendations, context)
    
    def _cascade_recommendation(self, user_id: str, n_recommendations: int,
                              context: Optional[Dict]) -> List[Tuple[str, float]]:
        """Cascade through recommenders, refining results at each stage."""
        # Start with first recommender
        initial_recommender = list(self.recommenders.keys())[0]
        recommendations = self.recommenders[initial_recommender].recommend(
            user_id, n_recommendations * 2
        )
        
        movie_ids = [movie_id for movie_id, _ in recommendations]
        
        # Refine with other recommenders
        for name, recommender in list(self.recommenders.items())[1:]:
            if not movie_ids:
                break
            
            # Get scores from this recommender for the candidate movies
            refined_scores = {}
            for movie_id in movie_ids:
                if hasattr(recommender, 'predict'):
                    score = recommender.predict(user_id, movie_id)
                    refined_scores[movie_id] = score
                elif hasattr(recommender, 'recommend_similar'):
                    # For content-based, use similarity to user's liked movies
                    score = self._get_content_score(recommender, user_id, movie_id)
                    refined_scores[movie_id] = score
            
            # Update scores (average with previous scores)
            weight = self.weights.get(name, 0.5)
            for i, (movie_id, score) in enumerate(recommendations):
                if movie_id in refined_scores:
                    new_score = (score * (1 - weight)) + (refined_scores[movie_id] * weight)
                    recommendations[i] = (movie_id, new_score)
        
        # Sort and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def _mixed_recommendation(self, user_id: str, n_recommendations: int,
                            context: Optional[Dict]) -> List[Tuple[str, float]]:
        """Mixed strategy: different recommenders for different recommendation slots."""
        recommendations_by_source = {}
        
        # Get recommendations from each source
        for name, recommender in self.recommenders.items():
            try:
                if hasattr(recommender, 'recommend'):
                    recs = recommender.recommend(user_id, n_recommendations)
                    recommendations_by_source[name] = recs
            except Exception as e:
                logger.warning(f"Recommender {name} failed in mixed mode: {str(e)}")
        
        # Mix recommendations ensuring diversity
        mixed_recommendations = []
        used_movies = set()
        
        # Round-robin through sources
        sources = list(recommendations_by_source.keys())
        source_idx = 0
        
        while len(mixed_recommendations) < n_recommendations and sources:
            source = sources[source_idx]
            recs = recommendations_by_source[source]
            
            # Find next unused movie from this source
            for movie_id, score in recs:
                if movie_id not in used_movies:
                    mixed_recommendations.append((movie_id, score))
                    used_movies.add(movie_id)
                    break
            
            # Move to next source
            source_idx = (source_idx + 1) % len(sources)
            
            # Remove sources that have no more unique recommendations
            sources = [s for s in sources 
                      if any(movie_id not in used_movies 
                            for movie_id, _ in recommendations_by_source.get(s, []))]
        
        return mixed_recommendations
    
    def _select_recommender(self, user_id: str, 
                          context: Optional[Dict]) -> str:
        """Select the most appropriate recommender based on context."""
        # Default: use weighted combination of all
        if not context:
            return 'weighted'
        
        # Check for cold start (new user or few interactions)
        if context.get('is_new_user', False) or context.get('interaction_count', 0) < 5:
            return next((name for name in self.recommenders.keys() 
                        if 'content' in name.lower()), list(self.recommenders.keys())[0])
        
        # Check for content-rich context
        if context.get('has_content_preferences', False):
            return next((name for name in self.recommenders.keys() 
                        if 'content' in name.lower()), list(self.recommenders.keys())[0])
        
        # Check for rich interaction history
        if context.get('interaction_count', 0) > 50:
            return next((name for name in self.recommenders.keys() 
                        if 'collaborative' in name.lower() or 'matrix' in name.lower()),
                       list(self.recommenders.keys())[0])
        
        # Default to first recommender
        return list(self.recommenders.keys())[0]
    
    def _get_content_score(self, content_recommender, user_id: str, 
                          movie_id: str) -> float:
        """Get content-based score for a movie."""
        # This is a simplified version - would need user's liked movies
        # For now, return a default score
        return 0.5
    
    def update_user_profile(self, user_id: str, 
                           interactions: List[Dict[str, Any]]) -> None:
        """
        Update user profile based on new interactions.
        
        Args:
            user_id: User ID
            interactions: List of new interactions
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'interaction_count': 0,
                'avg_rating': 0,
                'preferred_genres': defaultdict(int),
                'preferred_directors': defaultdict(int),
                'preferred_actors': defaultdict(int)
            }
        
        profile = self.user_profiles[user_id]
        
        for interaction in interactions:
            profile['interaction_count'] += 1
            
            # Update average rating
            if 'rating' in interaction:
                old_avg = profile['avg_rating']
                new_rating = interaction['rating']
                profile['avg_rating'] = (
                    (old_avg * (profile['interaction_count'] - 1) + new_rating) / 
                    profile['interaction_count']
                )
            
            # Update preferences (would need movie metadata)
            # This is simplified - in practice, would look up movie metadata
        
        # Update recommender weights based on user profile
        self._update_weights_for_user(user_id, profile)
    
    def _update_weights_for_user(self, user_id: str, profile: Dict) -> None:
        """Update recommender weights based on user profile."""
        # Adjust weights based on user characteristics
        if profile['interaction_count'] < 10:
            # New user: favor content-based
            self.weights = {
                name: 0.7 if 'content' in name.lower() else 0.3/len(self.recommenders)
                for name in self.recommenders.keys()
            }
        elif profile['interaction_count'] > 100:
            # Experienced user: favor collaborative
            self.weights = {
                name: 0.7 if 'collaborative' in name.lower() or 'matrix' in name.lower() 
                else 0.3/len(self.recommenders)
                for name in self.recommenders.keys()
            }
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
