#!/usr/bin/env python3
"""
Main application entry point for Movie Recommendation System.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline.extract import DataExtractor
from data_pipeline.transform import DataTransformer
from data_pipeline.load import DataLoader
from recommendation.collaborative import UserBasedCF, ItemBasedCF, MatrixFactorization
from recommendation.content_based import ContentBasedRecommender
from recommendation.hybrid import HybridRecommender
from storage.repositories import (
    MovieRepository, UserRepository, 
    InteractionRepository, RecommendationRepository
)
from api.endpoints import (
    RecommendationEndpoints, MovieEndpoints, UserEndpoints
)
from api.middleware import (
    RequestLoggingMiddleware, AuthenticationMiddleware,
    RateLimitingMiddleware, CORSMiddleware, ErrorHandlingMiddleware
)

import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/recommendation_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MovieRecommendationSystem:
    """Main Movie Recommendation System class."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()
        self.setup_logging()
        self.initialize_components()
        logger.info("Movie Recommendation System initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'data': {
                'input_folder': 'data/input',
                'processed_folder': 'data/processed',
                'models_folder': 'models'
            },
            'database': {
                'path': 'data/recommendation.db'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'api_keys': {'default_key': 'test_key_123'},
                'rate_limit': 100
            },
            'recommendation': {
                'default_algorithm': 'hybrid',
                'n_recommendations': 10
            }
        }
    
    def setup_logging(self):
        """Setup application logging."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
    
    def initialize_components(self):
        """Initialize all system components."""
        # Initialize data pipeline
        self.data_extractor = DataExtractor()
        self.data_transformer = DataTransformer()
        self.data_loader = DataLoader(self.config['data'])
        
        # Initialize repositories
        self.movie_repo = MovieRepository(self.config['database']['path'])
        self.user_repo = UserRepository(self.config['database']['path'])
        self.interaction_repo = InteractionRepository(self.config['database']['path'])
        self.recommendation_repo = RecommendationRepository(self.config['database']['path'])
        
        # Initialize recommendation algorithms
        self.collaborative_recommenders = {
            'user_based': UserBasedCF(),
            'item_based': ItemBasedCF(),
            'matrix_factorization': MatrixFactorization(n_factors=50)
        }
        
        self.content_recommender = ContentBasedRecommender()
        self.hybrid_recommender = HybridRecommender(
            recommenders={
                **self.collaborative_recommenders,
                'content_based': self.content_recommender
            },
            strategy='weighted'
        )
        
        # Flag to track if models are trained
        self.models_trained = False
    
    def load_data(self, input_folder: str = None):
        """
        Load and process data from input folder.
        
        Args:
            input_folder: Path to input folder (uses config if None)
        """
        try:
            if input_folder is None:
                input_folder = self.config['data']['input_folder']
            
            logger.info(f"Loading data from: {input_folder}")
            
            # Extract data
            raw_data = self.data_extractor.extract_all(input_folder)
            
            if not raw_data:
                logger.warning("No data extracted")
                return False
            
            # Transform data
            processed_data = {}
            
            if 'metadata' in raw_data:
                processed_data['metadata'] = self.data_transformer.clean_metadata(
                    raw_data['metadata']
                )
            
            if 'interactions' in raw_data:
                processed_data['interactions'] = self.data_transformer.clean_interactions(
                    raw_data['interactions']
                )
            
            # Create features
            if 'metadata' in processed_data and 'interactions' in processed_data:
                processed_data['user_features'] = self.data_transformer.create_user_features(
                    processed_data['interactions']
                )
                
                processed_data['movie_features'] = self.data_transformer.create_movie_features(
                    processed_data['metadata'],
                    processed_data['interactions']
                )
                
                processed_data['interaction_matrix'] = self.data_transformer.create_interaction_matrix(
                    processed_data['interactions'],
                    processed_data['user_features'],
                    processed_data['movie_features']
                )
            
            # Save processed data
            self.data_loader.save_to_disk(processed_data)
            
            # Save to database
            if 'metadata' in processed_data:
                self._save_movies_to_db(processed_data['metadata'])
            
            logger.info("Data loaded and processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def _save_movies_to_db(self, df_metadata: pd.DataFrame):
        """Save movies from DataFrame to database."""
        try:
            movies_data = df_metadata.to_dict('records')
            saved_count = self.movie_repo.save_batch_movies(movies_data)
            logger.info(f"Saved {saved_count} movies to database")
        except Exception as e:
            logger.error(f"Error saving movies to DB: {str(e)}")
    
    def train_models(self):
        """Train recommendation models."""
        try:
            logger.info("Training recommendation models")
            
            # Load processed data
            data = self.data_loader.load_from_disk([
                'metadata', 'interactions', 'interaction_matrix'
            ])
            
            if not data or 'interaction_matrix' not in data:
                logger.warning("No data available for training")
                return False
            
            # Train collaborative filtering models
            interaction_matrix = data['interaction_matrix']
            
            for name, recommender in self.collaborative_recommenders.items():
                logger.info(f"Training {name} model")
                recommender.fit(interaction_matrix)
                
                # Save model
                model_metadata = {
                    'type': 'collaborative',
                    'algorithm': name,
                    'trained_at': pd.Timestamp.now().isoformat(),
                    'data_shape': interaction_matrix.shape
                }
                self.data_loader.save_model(recommender, f"{name}_model", model_metadata)
            
            # Train content-based model
            if 'metadata' in data:
                logger.info("Training content-based model")
                self.content_recommender.fit(data['metadata'])
                
                model_metadata = {
                    'type': 'content_based',
                    'trained_at': pd.Timestamp.now().isoformat(),
                    'movie_count': len(data['metadata'])
                }
                self.data_loader.save_model(
                    self.content_recommender, 
                    'content_based_model', 
                    model_metadata
                )
            
            # Train hybrid model
            logger.info("Training hybrid model")
            training_data = {
                'interaction_matrix': interaction_matrix,
                'metadata': data.get('metadata')
            }
            self.hybrid_recommender.fit(training_data)
            
            model_metadata = {
                'type': 'hybrid',
                'strategy': self.hybrid_recommender.strategy,
                'weights': self.hybrid_recommender.weights,
                'trained_at': pd.Timestamp.now().isoformat()
            }
            self.data_loader.save_model(
                self.hybrid_recommender, 
                'hybrid_model', 
                model_metadata
            )
            
            self.models_trained = True
            logger.info("All models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10,
                           algorithm: str = None) -> List[Dict[str, Any]]:
        """
        Get recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            algorithm: Specific algorithm to use
            
        Returns:
            List of recommendations
        """
        try:
            if not self.models_trained:
                logger.warning("Models not trained, loading from disk")
                self._load_trained_models()
            
            if algorithm:
                # Use specific algorithm
                if algorithm in self.collaborative_recommenders:
                    raw_recommendations = self.collaborative_recommenders[algorithm].recommend(
                        user_id, n_recommendations
                    )
                elif algorithm == 'content_based':
                    # Get user's ratings for content-based
                    user_interactions = self.interaction_repo.get_user_interactions(user_id, limit=50)
                    user_ratings = {
                        interaction['movie_id']: interaction.get('rating', 3.0)
                        for interaction in user_interactions
                        if interaction.get('rating')
                    }
                    raw_recommendations = self.content_recommender.recommend_for_user(
                        user_ratings, n_recommendations
                    )
                else:
                    # Default to hybrid
                    raw_recommendations = self.hybrid_recommender.recommend(
                        user_id, n_recommendations
                    )
            else:
                # Use hybrid recommender by default
                raw_recommendations = self.hybrid_recommender.recommend(
                    user_id, n_recommendations
                )
            
            # Enhance recommendations with movie details
            recommendations = []
            for movie_id, score in raw_recommendations:
                movie_details = self.movie_repo.get_movie(movie_id)
                if movie_details:
                    recommendation = {
                        'movie_id': movie_id,
                        'title': movie_details.get('title', 'Unknown'),
                        'genres': movie_details.get('genres', []),
                        'score': float(score),
                        'poster_url': movie_details.get('poster_url'),
                        'release_year': movie_details.get('release_year')
                    }
                    recommendations.append(recommendation)
            
            # Save recommendations to database
            for rec in recommendations:
                rec_data = {
                    'user_id': user_id,
                    'movie_id': rec['movie_id'],
                    'algorithm': algorithm or 'hybrid',
                    'score': rec['score'],
                    'generated_at': pd.Timestamp.now().isoformat()
                }
                self.recommendation_repo.save_recommendation(rec_data)
            
            logger.info(f"Generated {len(recommendations)} recommendations for {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []
    
    def get_similar_movies(self, movie_id: str, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Get movies similar to a given movie.
        
        Args:
            movie_id: Movie ID
            n_recommendations: Number of similar movies
            
        Returns:
            List of similar movies
        """
        try:
            if not self.models_trained:
                self._load_trained_models()
            
            # Get similar movies from content-based recommender
            raw_similar = self.content_recommender.recommend_similar(
                movie_id, n_recommendations
            )
            
            # Enhance with movie details
            similar_movies = []
            for similar_id, similarity in raw_similar:
                movie_details = self.movie_repo.get_movie(similar_id)
                if movie_details:
                    similar_movie = {
                        'movie_id': similar_id,
                        'title': movie_details.get('title', 'Unknown'),
                        'genres': movie_details.get('genres', []),
                        'similarity_score': float(similarity),
                        'poster_url': movie_details.get('poster_url'),
                        'release_year': movie_details.get('release_year')
                    }
                    similar_movies.append(similar_movie)
            
            return similar_movies
            
        except Exception as e:
            logger.error(f"Error getting similar movies: {str(e)}")
            return []
    
    def record_interaction(self, interaction_data: Dict[str, Any]) -> bool:
        """
        Record a user-movie interaction.
        
        Args:
            interaction_data: Interaction data
            
        Returns:
            True if successful
        """
        try:
            # Save interaction to database
            interaction_id = self.interaction_repo.save_interaction(interaction_data)
            
            if interaction_id > 0:
                # Update user statistics
                self.user_repo.get_user_statistics(interaction_data['user_id'])
                
                # Update hybrid recommender user profile
                self.hybrid_recommender.update_user_profile(
                    interaction_data['user_id'],
                    [interaction_data]
                )
                
                logger.info(f"Recorded interaction #{interaction_id}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error recording interaction: {str(e)}")
            return False
    
    def get_available_algorithms(self) -> List[Dict[str, Any]]:
        """Get list of available recommendation algorithms."""
        algorithms = [
            {
                'name': 'user_based',
                'type': 'collaborative',
                'description': 'User-based collaborative filtering'
            },
            {
                'name': 'item_based',
                'type': 'collaborative',
                'description': 'Item-based collaborative filtering'
            },
            {
                'name': 'matrix_factorization',
                'type': 'collaborative',
                'description': 'Matrix factorization (SVD)'
            },
            {
                'name': 'content_based',
                'type': 'content',
                'description': 'Content-based recommendation'
            },
            {
                'name': 'hybrid',
                'type': 'hybrid',
                'description': 'Hybrid approach combining multiple methods'
            }
        ]
        return algorithms
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            'system': {
                'models_trained': self.models_trained,
                'recommenders_available': len(self.get_available_algorithms())
            },
            'database': {
                'movies': 0,
                'users': 0,
                'interactions': 0,
                'recommendations': 0
            }
        }
        
        # Get database statistics
        try:
            # Count movies
            result = self.movie_repo.execute_query("SELECT COUNT(*) FROM movies")
            stats['database']['movies'] = result[0][0] if result else 0
            
            # Count users
            result = self.user_repo.execute_query("SELECT COUNT(*) FROM users")
            stats['database']['users'] = result[0][0] if result else 0
            
            # Count interactions
            result = self.interaction_repo.execute_query("SELECT COUNT(*) FROM interactions")
            stats['database']['interactions'] = result[0][0] if result else 0
            
            # Count recommendations
            result = self.recommendation_repo.execute_query("SELECT COUNT(*) FROM recommendations")
            stats['database']['recommendations'] = result[0][0] if result else 0
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
        
        return stats
    
    def _load_trained_models(self):
        """Load trained models from disk."""
        try:
            logger.info("Loading trained models from disk")
            
            # Load collaborative models
            for name in self.collaborative_recommenders.keys():
                try:
                    model_info = self.data_loader.load_model(f"{name}_model")
                    self.collaborative_recommenders[name] = model_info['model']
                except FileNotFoundError:
                    logger.warning(f"Model not found: {name}_model")
            
            # Load content-based model
            try:
                model_info = self.data_loader.load_model('content_based_model')
                self.content_recommender = model_info['model']
            except FileNotFoundError:
                logger.warning("Content-based model not found")
            
            # Load hybrid model
            try:
                model_info = self.data_loader.load_model('hybrid_model')
                self.hybrid_recommender = model_info['model']
            except FileNotFoundError:
                logger.warning("Hybrid model not found")
            
            self.models_trained = True
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_trained = False
    
    def create_api_app(self) -> FastAPI:
        """Create FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting Movie Recommendation System API")
            yield
            # Shutdown
            logger.info("Shutting down Movie Recommendation System API")
        
        # Create FastAPI app
        app = FastAPI(
            title="Movie Recommendation System API",
            description="API for movie recommendations using collaborative filtering, content-based, and hybrid approaches",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Initialize API endpoints
        recommendation_endpoints = RecommendationEndpoints(self)
        movie_endpoints = MovieEndpoints(self)
        user_endpoints = UserEndpoints(self)
        
        # Include routers
        app.include_router(recommendation_endpoints.router, prefix="/api/v1", tags=["recommendations"])
        app.include_router(movie_endpoints.router, prefix="/api/v1", tags=["movies"])
        app.include_router(user_endpoints.router, prefix="/api/v1", tags=["users"])
        
        # Add middleware
        # Note: In FastAPI, middleware is added differently
        # This would be configured in the actual uvicorn run
        
        return app

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data loading command
    load_parser = subparsers.add_parser("load-data", help="Load and process data")
    load_parser.add_argument("--input", "-i", default="data/input", 
                           help="Input folder path")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train recommendation models")
    
    # API server command
    api_parser = subparsers.add_parser("serve", help="Start API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind to")
    
    # Recommendation command
    rec_parser = subparsers.add_parser("recommend", help="Get recommendations")
    rec_parser.add_argument("--user", "-u", required=True, help="User ID")
    rec_parser.add_argument("--count", "-c", type=int, default=10, 
                          help="Number of recommendations")
    rec_parser.add_argument("--algorithm", "-a", default="hybrid",
                          help="Recommendation algorithm to use")
    
    args = parser.parse_args()
    
    # Initialize system
    system = MovieRecommendationSystem()
    
    if args.command == "load-data":
        success = system.load_data(args.input)
        sys.exit(0 if success else 1)
    
    elif args.command == "train":
        success = system.train_models()
        sys.exit(0 if success else 1)
    
    elif args.command == "serve":
        app = system.create_api_app()
        uvicorn.run(app, host=args.host, port=args.port)
    
    elif args.command == "recommend":
        recommendations = system.get_recommendations(
            args.user, args.count, args.algorithm
        )
        
        if recommendations:
            print(f"\nRecommendations for user '{args.user}':")
            print("-" * 80)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i:2}. {rec['title']} ({rec.get('release_year', 'N/A')})")
                print(f"    ID: {rec['movie_id']}")
                print(f"    Score: {rec['score']:.3f}")
                print(f"    Genres: {', '.join(rec.get('genres', []))}")
                print()
        else:
            print(f"No recommendations found for user '{args.user}'")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
