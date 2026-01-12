# movie-recommendation-system/api/schemas/__init__.py
"""
Pydantic schemas for Movie Recommendation System API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class RecommendationAlgorithm(str, Enum):
    """Available recommendation algorithms."""
    USER_BASED = "user_based"
    ITEM_BASED = "item_based"
    MATRIX_FACTORIZATION = "matrix_factorization"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"

class RecommendationRequest(BaseModel):
    """Request schema for getting recommendations."""
    user_id: str = Field(..., description="User ID")
    n_recommendations: int = Field(10, ge=1, le=100, 
                                  description="Number of recommendations")
    algorithm: Optional[RecommendationAlgorithm] = Field(
        RecommendationAlgorithm.HYBRID,
        description="Recommendation algorithm to use"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context for recommendations"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "n_recommendations": 10,
                "algorithm": "hybrid",
                "context": {"device": "mobile", "time_of_day": "evening"}
            }
        }

class RecommendationResponse(BaseModel):
    """Response schema for recommendations."""
    user_id: str = Field(..., description="User ID")
    recommendations: List[Dict[str, Any]] = Field(
        ...,
        description="List of recommended movies with scores"
    )
    count: int = Field(..., description="Number of recommendations")
    algorithm: Optional[str] = Field(None, description="Algorithm used")
    generated_at: datetime = Field(default_factory=datetime.utcnow,
                                  description="Timestamp of generation")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "recommendations": [
                    {"movie_id": "tt1234567", "title": "Sample Movie", 
                     "score": 4.8, "reason": "Similar to movies you liked"},
                    {"movie_id": "tt2345678", "title": "Another Movie", 
                     "score": 4.5, "reason": "Popular with similar users"}
                ],
                "count": 2,
                "algorithm": "hybrid",
                "generated_at": "2024-01-15T10:30:00Z"
            }
        }

class SimilarMoviesRequest(BaseModel):
    """Request schema for getting similar movies."""
    movie_id: str = Field(..., description="Movie ID")
    n_recommendations: int = Field(10, ge=1, le=50,
                                  description="Number of similar movies")
    
    class Config:
        schema_extra = {
            "example": {
                "movie_id": "tt1234567",
                "n_recommendations": 10
            }
        }

class SimilarMoviesResponse(BaseModel):
    """Response schema for similar movies."""
    movie_id: str = Field(..., description="Movie ID")
    similar_movies: List[Dict[str, Any]] = Field(
        ...,
        description="List of similar movies with similarity scores"
    )
    count: int = Field(..., description="Number of similar movies")
    
    class Config:
        schema_extra = {
            "example": {
                "movie_id": "tt1234567",
                "similar_movies": [
                    {"movie_id": "tt2345678", "title": "Similar Movie", 
                     "similarity_score": 0.92},
                    {"movie_id": "tt3456789", "title": "Another Similar Movie", 
                     "similarity_score": 0.87}
                ],
                "count": 2
            }
        }

class InteractionType(str, Enum):
    """Types of user-movie interactions."""
    RATING = "rating"
    WATCH = "watch"
    CLICK = "click"
    SAVE = "save"
    SHARE = "share"

class InteractionRequest(BaseModel):
    """Request schema for recording interactions."""
    user_id: str = Field(..., description="User ID")
    movie_id: str = Field(..., description="Movie ID")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Rating (0-5)")
    duration: Optional[float] = Field(None, ge=0, description="Watch duration in minutes")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow,
                                         description="Interaction timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @validator('rating')
    def validate_rating_for_type(cls, v, values):
        """Validate that rating is provided for rating interactions."""
        if values.get('interaction_type') == InteractionType.RATING and v is None:
            raise ValueError('Rating is required for rating interactions')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "movie_id": "tt1234567",
                "interaction_type": "rating",
                "rating": 4.5,
                "duration": 120,
                "timestamp": "2024-01-15T10:30:00Z",
                "metadata": {"device": "web", "location": "US"}
            }
        }

class InteractionResponse(BaseModel):
    """Response schema for interaction recording."""
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Response message")
    interaction_id: Optional[str] = Field(None, description="Generated interaction ID")
    recorded_at: datetime = Field(default_factory=datetime.utcnow,
                                 description="Timestamp of recording")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Interaction recorded",
                "interaction_id": "interaction_123",
                "recorded_at": "2024-01-15T10:30:00Z"
            }
        }

class MovieSearchRequest(BaseModel):
    """Request schema for searching movies."""
    query: Optional[str] = Field(None, description="Search query")
    genres: Optional[List[str]] = Field(None, description="Filter by genres")
    year_min: Optional[int] = Field(None, ge=1900, le=2100, 
                                   description="Minimum release year")
    year_max: Optional[int] = Field(None, ge=1900, le=2100,
                                   description="Maximum release year")
    rating_min: Optional[float] = Field(None, ge=0, le=5,
                                       description="Minimum rating")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Pagination offset")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "action",
                "genres": ["Action", "Adventure"],
                "year_min": 2010,
                "year_max": 2023,
                "rating_min": 3.5,
                "limit": 20,
                "offset": 0
            }
        }

class MovieResponse(BaseModel):
    """Response schema for movie information."""
    movie_id: str = Field(..., description="Movie ID")
    title: str = Field(..., description="Movie title")
    genres: List[str] = Field(..., description="Movie genres")
    release_year: Optional[int] = Field(None, description="Release year")
    description: Optional[str] = Field(None, description="Movie description")
    directors: Optional[List[str]] = Field(None, description="Directors")
    actors: Optional[List[str]] = Field(None, description="Main actors")
    imdb_rating: Optional[float] = Field(None, ge=0, le=10, description="IMDB rating")
    duration_minutes: Optional[int] = Field(None, ge=0, description="Duration in minutes")
    poster_url: Optional[str] = Field(None, description="URL to poster image")
    trailer_url: Optional[str] = Field(None, description="URL to trailer")
    
    class Config:
        schema_extra = {
            "example": {
                "movie_id": "tt1234567",
                "title": "Sample Movie",
                "genres": ["Action", "Adventure"],
                "release_year": 2023,
                "description": "A sample movie description",
                "directors": ["Director One"],
                "actors": ["Actor One", "Actor Two"],
                "imdb_rating": 7.5,
                "duration_minutes": 120,
                "poster_url": "https://example.com/poster.jpg",
                "trailer_url": "https://example.com/trailer.mp4"
            }
        }

class MovieSearchResponse(BaseModel):
    """Response schema for movie search."""
    movies: List[MovieResponse] = Field(..., description="List of movies")
    count: int = Field(..., description="Number of movies returned")
    total: Optional[int] = Field(None, description="Total matching movies")
    limit: int = Field(..., description="Maximum results per page")
    offset: int = Field(..., description="Pagination offset")
    
    class Config:
        schema_extra = {
            "example": {
                "movies": [
                    {
                        "movie_id": "tt1234567",
                        "title": "Sample Movie",
                        "genres": ["Action", "Adventure"]
                    }
                ],
                "count": 1,
                "total": 100,
                "limit": 20,
                "offset": 0
            }
        }

class UserProfileRequest(BaseModel):
    """Request schema for updating user profile."""
    preferences: Dict[str, Any] = Field(..., description="User preferences")
    
    class Config:
        schema_extra = {
            "example": {
                "preferences": {
                    "favorite_genres": ["Action", "Sci-Fi"],
                    "preferred_languages": ["English", "Spanish"],
                    "content_ratings": ["PG-13", "R"],
                    "watch_history_enabled": True
                }
            }
        }

class UserProfileResponse(BaseModel):
    """Response schema for user profile."""
    user_id: str = Field(..., description="User ID")
    preferences: Dict[str, Any] = Field(..., description="User preferences")
    statistics: Dict[str, Any] = Field(..., description="User statistics")
    created_at: datetime = Field(..., description="Profile creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "preferences": {
                    "favorite_genres": ["Action", "Sci-Fi"]
                },
                "statistics": {
                    "movies_watched": 42,
                    "average_rating": 3.8,
                    "favorite_genre": "Action"
                },
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }

class BatchRecommendationRequest(BaseModel):
    """Request schema for batch recommendations."""
    user_ids: List[str] = Field(..., min_items=1, max_items=100,
                               description="List of user IDs")
    n_recommendations: int = Field(10, ge=1, le=20,
                                  description="Recommendations per user")
    algorithm: Optional[RecommendationAlgorithm] = Field(
        RecommendationAlgorithm.HYBRID,
        description="Recommendation algorithm to use"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "user_ids": ["user_123", "user_456"],
                "n_recommendations": 10,
                "algorithm": "hybrid"
            }
        }

class BatchRecommendationResponse(BaseModel):
    """Response schema for batch recommendations."""
    batch_recommendations: Dict[str, List[Dict[str, Any]]] = Field(
        ...,
        description="Recommendations by user ID"
    )
    user_count: int = Field(..., description="Number of users processed")
    generated_at: datetime = Field(default_factory=datetime.utcnow,
                                  description="Timestamp of generation")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_recommendations": {
                    "user_123": [
                        {"movie_id": "tt1234567", "score": 4.8}
                    ]
                },
                "user_count": 1,
                "generated_at": "2024-01-15T10:30:00Z"
            }
        }

class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow,
                               description="Current timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "movie-recommendation-api",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class ErrorResponse(BaseModel):
    """Response schema for errors."""
    status: str = Field(..., description="Error status")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow,
                               description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "message": "User not found",
                "error_code": "USER_NOT_FOUND",
                "details": {"user_id": "user_999"},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }