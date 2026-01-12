# movie-recommendation-system/api/endpoints/__init__.py
"""
API endpoints for Movie Recommendation System.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RecommendationEndpoints:
    """API endpoints for recommendations."""
    
    def __init__(self, recommender_system):
        self.recommender_system = recommender_system
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.router.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "service": "movie-recommendation-api"}
        
        @self.router.get("/recommendations/{user_id}")
        async def get_recommendations(
            user_id: str,
            n_recommendations: int = 10,
            algorithm: Optional[str] = None
        ):
            """
            Get movie recommendations for a user.
            
            Args:
                user_id: User ID
                n_recommendations: Number of recommendations (default: 10)
                algorithm: Specific algorithm to use (optional)
            
            Returns:
                List of recommendations
            """
            try:
                logger.info(f"Getting recommendations for user: {user_id}")
                
                if algorithm:
                    # Use specific algorithm if specified
                    recommendations = self.recommender_system.get_recommendations(
                        user_id=user_id,
                        n_recommendations=n_recommendations,
                        algorithm=algorithm
                    )
                else:
                    # Use default/hybrid approach
                    recommendations = self.recommender_system.get_recommendations(
                        user_id=user_id,
                        n_recommendations=n_recommendations
                    )
                
                return {
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "count": len(recommendations)
                }
                
            except Exception as e:
                logger.error(f"Error getting recommendations: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/recommendations/similar/{movie_id}")
        async def get_similar_movies(
            movie_id: str,
            n_recommendations: int = 10
        ):
            """
            Get movies similar to a given movie.
            
            Args:
                movie_id: Movie ID
                n_recommendations: Number of similar movies (default: 10)
            
            Returns:
                List of similar movies
            """
            try:
                logger.info(f"Getting similar movies for: {movie_id}")
                
                similar_movies = self.recommender_system.get_similar_movies(
                    movie_id=movie_id,
                    n_recommendations=n_recommendations
                )
                
                return {
                    "movie_id": movie_id,
                    "similar_movies": similar_movies,
                    "count": len(similar_movies)
                }
                
            except Exception as e:
                logger.error(f"Error getting similar movies: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/recommendations/batch")
        async def get_batch_recommendations(
            user_ids: List[str],
            n_recommendations: int = 10
        ):
            """
            Get recommendations for multiple users.
            
            Args:
                user_ids: List of user IDs
                n_recommendations: Number of recommendations per user
            
            Returns:
                Dictionary of recommendations by user
            """
            try:
                logger.info(f"Getting batch recommendations for {len(user_ids)} users")
                
                batch_recommendations = {}
                for user_id in user_ids:
                    recommendations = self.recommender_system.get_recommendations(
                        user_id=user_id,
                        n_recommendations=n_recommendations
                    )
                    batch_recommendations[user_id] = recommendations
                
                return {
                    "batch_recommendations": batch_recommendations,
                    "user_count": len(user_ids)
                }
                
            except Exception as e:
                logger.error(f"Error in batch recommendations: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.post("/interactions")
        async def record_interaction(interaction: Dict[str, Any]):
            """
            Record a user-movie interaction.
            
            Args:
                interaction: Interaction data including user_id, movie_id, rating, etc.
            
            Returns:
                Confirmation of recorded interaction
            """
            try:
                logger.info(f"Recording interaction: {interaction}")
                
                # Validate required fields
                required_fields = ['user_id', 'movie_id']
                for field in required_fields:
                    if field not in interaction:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Missing required field: {field}"
                        )
                
                # Record interaction in the system
                success = self.recommender_system.record_interaction(interaction)
                
                if success:
                    return {
                        "status": "success",
                        "message": "Interaction recorded",
                        "interaction": interaction
                    }
                else:
                    raise HTTPException(
                        status_code=500, 
                        detail="Failed to record interaction"
                    )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error recording interaction: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/algorithms")
        async def get_available_algorithms():
            """
            Get list of available recommendation algorithms.
            
            Returns:
                List of algorithm information
            """
            try:
                algorithms = self.recommender_system.get_available_algorithms()
                return {
                    "algorithms": algorithms,
                    "count": len(algorithms)
                }
            except Exception as e:
                logger.error(f"Error getting algorithms: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/stats")
        async def get_system_stats():
            """
            Get system statistics.
            
            Returns:
                System statistics
            """
            try:
                stats = self.recommender_system.get_statistics()
                return stats
            except Exception as e:
                logger.error(f"Error getting stats: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

class MovieEndpoints:
    """API endpoints for movie information."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.router.get("/movies/{movie_id}")
        async def get_movie_details(movie_id: str):
            """
            Get detailed information about a movie.
            
            Args:
                movie_id: Movie ID
            
            Returns:
                Movie details
            """
            try:
                movie_details = self.data_manager.get_movie_details(movie_id)
                
                if not movie_details:
                    raise HTTPException(status_code=404, detail="Movie not found")
                
                return movie_details
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting movie details: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/movies")
        async def search_movies(
            query: Optional[str] = None,
            genre: Optional[str] = None,
            year_min: Optional[int] = None,
            year_max: Optional[int] = None,
            limit: int = 20,
            offset: int = 0
        ):
            """
            Search for movies with filters.
            
            Args:
                query: Search query for title/description
                genre: Filter by genre
                year_min: Minimum release year
                year_max: Maximum release year
                limit: Maximum number of results
                offset: Pagination offset
            
            Returns:
                List of matching movies
            """
            try:
                filters = {
                    'query': query,
                    'genre': genre,
                    'year_min': year_min,
                    'year_max': year_max
                }
                
                movies = self.data_manager.search_movies(filters, limit, offset)
                
                return {
                    "movies": movies,
                    "count": len(movies),
                    "limit": limit,
                    "offset": offset
                }
                
            except Exception as e:
                logger.error(f"Error searching movies: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/movies/popular")
        async def get_popular_movies(
            limit: int = 20,
            time_period: str = "month"  # day, week, month, year, all_time
        ):
            """
            Get popular movies based on interactions.
            
            Args:
                limit: Maximum number of results
                time_period: Time period for popularity calculation
            
            Returns:
                List of popular movies
            """
            try:
                popular_movies = self.data_manager.get_popular_movies(limit, time_period)
                
                return {
                    "popular_movies": popular_movies,
                    "count": len(popular_movies),
                    "time_period": time_period
                }
                
            except Exception as e:
                logger.error(f"Error getting popular movies: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/genres")
        async def get_all_genres():
            """
            Get all available movie genres.
            
            Returns:
                List of genres with counts
            """
            try:
                genres = self.data_manager.get_all_genres()
                return {
                    "genres": genres,
                    "count": len(genres)
                }
            except Exception as e:
                logger.error(f"Error getting genres: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

class UserEndpoints:
    """API endpoints for user information."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.router.get("/users/{user_id}/profile")
        async def get_user_profile(user_id: str):
            """
            Get user profile and preferences.
            
            Args:
                user_id: User ID
            
            Returns:
                User profile information
            """
            try:
                profile = self.data_manager.get_user_profile(user_id)
                
                if not profile:
                    raise HTTPException(status_code=404, detail="User not found")
                
                return profile
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting user profile: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/users/{user_id}/history")
        async def get_user_history(
            user_id: str,
            limit: int = 50,
            offset: int = 0
        ):
            """
            Get user's interaction history.
            
            Args:
                user_id: User ID
                limit: Maximum number of results
                offset: Pagination offset
            
            Returns:
                User interaction history
            """
            try:
                history = self.data_manager.get_user_history(user_id, limit, offset)
                
                return {
                    "user_id": user_id,
                    "history": history,
                    "count": len(history),
                    "limit": limit,
                    "offset": offset
                }
                
            except Exception as e:
                logger.error(f"Error getting user history: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.put("/users/{user_id}/preferences")
        async def update_user_preferences(
            user_id: str,
            preferences: Dict[str, Any]
        ):
            """
            Update user preferences.
            
            Args:
                user_id: User ID
                preferences: Updated preferences
            
            Returns:
                Updated user profile
            """
            try:
                updated_profile = self.data_manager.update_user_preferences(
                    user_id, preferences
                )
                
                return {
                    "status": "success",
                    "message": "Preferences updated",
                    "profile": updated_profile
                }
                
            except Exception as e:
                logger.error(f"Error updating preferences: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.delete("/users/{user_id}")
        async def delete_user_data(user_id: str):
            """
            Delete all user data (GDPR compliance).
            
            Args:
                user_id: User ID
            
            Returns:
                Confirmation of deletion
            """
            try:
                success = self.data_manager.delete_user_data(user_id)
                
                if success:
                    return {
                        "status": "success",
                        "message": f"User data for {user_id} deleted"
                    }
                else:
                    raise HTTPException(
                        status_code=500, 
                        detail="Failed to delete user data"
                    )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting user data: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))