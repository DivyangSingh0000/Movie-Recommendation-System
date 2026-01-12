"""
Repository pattern implementations for data access.
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class BaseRepository:
    """Base repository class with common database operations."""
    
    def __init__(self, db_path: str = "data/recommendation.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create movies table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS movies (
                        movie_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        genres TEXT,
                        release_year INTEGER,
                        description TEXT,
                        directors TEXT,
                        actors TEXT,
                        imdb_rating REAL,
                        duration_minutes INTEGER,
                        language TEXT,
                        country TEXT,
                        poster_url TEXT,
                        trailer_url TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create users table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        preferences TEXT,
                        statistics TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create interactions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        movie_id TEXT NOT NULL,
                        interaction_type TEXT NOT NULL,
                        rating REAL,
                        duration REAL,
                        timestamp TIMESTAMP,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id),
                        FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
                    )
                """)
                
                # Create recommendations table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS recommendations (
                        recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        movie_id TEXT NOT NULL,
                        algorithm TEXT NOT NULL,
                        score REAL NOT NULL,
                        generated_at TIMESTAMP NOT NULL,
                        served_at TIMESTAMP,
                        clicked BOOLEAN DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id),
                        FOREIGN KEY (movie_id) REFERENCES movies (movie_id)
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_interactions_user_movie 
                    ON interactions (user_id, movie_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_interactions_timestamp 
                    ON interactions (timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_recommendations_user 
                    ON recommendations (user_id, generated_at)
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Tuple]:
        """Execute a query and return results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an update query and return affected rows."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error executing update: {str(e)}")
            raise

class MovieRepository(BaseRepository):
    """Repository for movie data operations."""
    
    def save_movie(self, movie_data: Dict[str, Any]) -> bool:
        """Save or update a movie in the database."""
        try:
            # Prepare data
            genres = json.dumps(movie_data.get('genres', []))
            directors = json.dumps(movie_data.get('directors', []))
            actors = json.dumps(movie_data.get('actors', []))
            
            query = """
                INSERT OR REPLACE INTO movies 
                (movie_id, title, genres, release_year, description, 
                 directors, actors, imdb_rating, duration_minutes, 
                 language, country, poster_url, trailer_url, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            
            params = (
                movie_data['movie_id'],
                movie_data.get('title', ''),
                genres,
                movie_data.get('release_year'),
                movie_data.get('description', ''),
                directors,
                actors,
                movie_data.get('imdb_rating'),
                movie_data.get('duration_minutes'),
                movie_data.get('language', ''),
                movie_data.get('country', ''),
                movie_data.get('poster_url', ''),
                movie_data.get('trailer_url', '')
            )
            
            self.execute_update(query, params)
            logger.info(f"Saved movie: {movie_data['movie_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving movie: {str(e)}")
            return False
    
    def get_movie(self, movie_id: str) -> Optional[Dict[str, Any]]:
        """Get a movie by ID."""
        try:
            query = "SELECT * FROM movies WHERE movie_id = ?"
            result = self.execute_query(query, (movie_id,))
            
            if not result:
                return None
            
            row = result[0]
            movie = {
                'movie_id': row[0],
                'title': row[1],
                'genres': json.loads(row[2]) if row[2] else [],
                'release_year': row[3],
                'description': row[4],
                'directors': json.loads(row[5]) if row[5] else [],
                'actors': json.loads(row[6]) if row[6] else [],
                'imdb_rating': row[7],
                'duration_minutes': row[8],
                'language': row[9],
                'country': row[10],
                'poster_url': row[11],
                'trailer_url': row[12],
                'created_at': row[13],
                'updated_at': row[14]
            }
            
            return movie
            
        except Exception as e:
            logger.error(f"Error getting movie: {str(e)}")
            return None
    
    def search_movies(self, filters: Dict[str, Any], 
                     limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Search movies with filters."""
        try:
            query = "SELECT * FROM movies WHERE 1=1"
            params = []
            
            # Apply filters
            if 'query' in filters and filters['query']:
                query += " AND (title LIKE ? OR description LIKE ?)"
                search_term = f"%{filters['query']}%"
                params.extend([search_term, search_term])
            
            if 'genres' in filters and filters['genres']:
                # This is simplified - in production, you'd need better genre search
                query += " AND genres LIKE ?"
                genre_term = f"%{filters['genres'][0]}%"
                params.append(genre_term)
            
            if 'year_min' in filters:
                query += " AND release_year >= ?"
                params.append(filters['year_min'])
            
            if 'year_max' in filters:
                query += " AND release_year <= ?"
                params.append(filters['year_max'])
            
            if 'rating_min' in filters:
                query += " AND imdb_rating >= ?"
                params.append(filters['rating_min'])
            
            # Add pagination
            query += " ORDER BY imdb_rating DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            results = self.execute_query(query, params)
            movies = []
            
            for row in results:
                movie = {
                    'movie_id': row[0],
                    'title': row[1],
                    'genres': json.loads(row[2]) if row[2] else [],
                    'release_year': row[3],
                    'description': row[4],
                    'directors': json.loads(row[5]) if row[5] else [],
                    'actors': json.loads(row[6]) if row[6] else [],
                    'imdb_rating': row[7],
                    'duration_minutes': row[8],
                    'language': row[9],
                    'country': row[10],
                    'poster_url': row[11],
                    'trailer_url': row[12]
                }
                movies.append(movie)
            
            return movies
            
        except Exception as e:
            logger.error(f"Error searching movies: {str(e)}")
            return []
    
    def get_popular_movies(self, limit: int = 20, 
                          time_period: str = "month") -> List[Dict[str, Any]]:
        """Get popular movies based on interactions."""
        try:
            # Calculate time period
            time_filters = {
                "day": "1 day",
                "week": "7 days",
                "month": "30 days",
                "year": "365 days",
                "all_time": None
            }
            
            time_filter = time_filters.get(time_period, "30 days")
            
            query = """
                SELECT m.*, COUNT(i.interaction_id) as interaction_count
                FROM movies m
                LEFT JOIN interactions i ON m.movie_id = i.movie_id
            """
            
            if time_filter:
                query += f" WHERE i.timestamp >= datetime('now', '-{time_filter}')"
            
            query += """
                GROUP BY m.movie_id
                ORDER BY interaction_count DESC, m.imdb_rating DESC
                LIMIT ?
            """
            
            results = self.execute_query(query, (limit,))
            movies = []
            
            for row in results:
                movie = {
                    'movie_id': row[0],
                    'title': row[1],
                    'genres': json.loads(row[2]) if row[2] else [],
                    'release_year': row[3],
                    'description': row[4],
                    'directors': json.loads(row[5]) if row[5] else [],
                    'actors': json.loads(row[6]) if row[6] else [],
                    'imdb_rating': row[7],
                    'duration_minutes': row[8],
                    'language': row[9],
                    'country': row[10],
                    'poster_url': row[11],
                    'trailer_url': row[12],
                    'interaction_count': row[15] or 0
                }
                movies.append(movie)
            
            return movies
            
        except Exception as e:
            logger.error(f"Error getting popular movies: {str(e)}")
            return []
    
    def get_all_genres(self) -> List[Dict[str, Any]]:
        """Get all genres with movie counts."""
        try:
            query = """
                SELECT genre, COUNT(*) as movie_count
                FROM (
                    SELECT json_each.value as genre
                    FROM movies, json_each(movies.genres)
                )
                GROUP BY genre
                ORDER BY movie_count DESC
            """
            
            results = self.execute_query(query)
            genres = []
            
            for row in results:
                genres.append({
                    'genre': row[0],
                    'movie_count': row[1]
                })
            
            return genres
            
        except Exception as e:
            logger.error(f"Error getting genres: {str(e)}")
            return []
    
    def save_batch_movies(self, movies_data: List[Dict[str, Any]]) -> int:
        """Save multiple movies in batch."""
        try:
            saved_count = 0
            for movie_data in movies_data:
                if self.save_movie(movie_data):
                    saved_count += 1
            
            logger.info(f"Saved {saved_count} movies in batch")
            return saved_count
            
        except Exception as e:
            logger.error(f"Error saving batch movies: {str(e)}")
            return 0

class UserRepository(BaseRepository):
    """Repository for user data operations."""
    
    def save_user(self, user_id: str, preferences: Dict[str, Any] = None) -> bool:
        """Save or update a user in the database."""
        try:
            preferences_json = json.dumps(preferences or {})
            
            query = """
                INSERT OR REPLACE INTO users 
                (user_id, preferences, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """
            
            self.execute_update(query, (user_id, preferences_json))
            logger.info(f"Saved user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving user: {str(e)}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID."""
        try:
            query = "SELECT * FROM users WHERE user_id = ?"
            result = self.execute_query(query, (user_id,))
            
            if not result:
                return None
            
            row = result[0]
            user = {
                'user_id': row[0],
                'preferences': json.loads(row[1]) if row[1] else {},
                'statistics': json.loads(row[2]) if row[2] else {},
                'created_at': row[3],
                'updated_at': row[4]
            }
            
            return user
            
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            return None
    
    def update_user_preferences(self, user_id: str, 
                               preferences: Dict[str, Any]) -> bool:
        """Update user preferences."""
        try:
            current_user = self.get_user(user_id)
            if not current_user:
                # Create user if doesn't exist
                return self.save_user(user_id, preferences)
            
            # Merge with existing preferences
            current_prefs = current_user.get('preferences', {})
            merged_prefs = {**current_prefs, **preferences}
            
            query = """
                UPDATE users 
                SET preferences = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """
            
            self.execute_update(query, (json.dumps(merged_prefs), user_id))
            logger.info(f"Updated preferences for user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating preferences: {str(e)}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user and their data."""
        try:
            # Delete user interactions first (foreign key constraint)
            self.execute_update(
                "DELETE FROM interactions WHERE user_id = ?", 
                (user_id,)
            )
            
            # Delete user recommendations
            self.execute_update(
                "DELETE FROM recommendations WHERE user_id = ?", 
                (user_id,)
            )
            
            # Delete user
            self.execute_update(
                "DELETE FROM users WHERE user_id = ?", 
                (user_id,)
            )
            
            logger.info(f"Deleted user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Calculate user statistics."""
        try:
            stats = {
                'interaction_count': 0,
                'average_rating': 0,
                'rated_movies': 0,
                'watched_movies': 0,
                'favorite_genres': []
            }
            
            # Get interaction count
            query = "SELECT COUNT(*) FROM interactions WHERE user_id = ?"
            result = self.execute_query(query, (user_id,))
            stats['interaction_count'] = result[0][0] if result else 0
            
            # Get average rating
            query = """
                SELECT AVG(rating) 
                FROM interactions 
                WHERE user_id = ? AND rating IS NOT NULL
            """
            result = self.execute_query(query, (user_id,))
            stats['average_rating'] = float(result[0][0] or 0) if result else 0
            
            # Get rated movies count
            query = """
                SELECT COUNT(DISTINCT movie_id) 
                FROM interactions 
                WHERE user_id = ? AND rating IS NOT NULL
            """
            result = self.execute_query(query, (user_id,))
            stats['rated_movies'] = result[0][0] if result else 0
            
            # Get watched movies count
            query = """
                SELECT COUNT(DISTINCT movie_id) 
                FROM interactions 
                WHERE user_id = ? AND interaction_type = 'watch'
            """
            result = self.execute_query(query, (user_id,))
            stats['watched_movies'] = result[0][0] if result else 0
            
            # Update statistics in user record
            if self.get_user(user_id):
                query = """
                    UPDATE users 
                    SET statistics = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """
                self.execute_update(query, (json.dumps(stats), user_id))
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating user statistics: {str(e)}")
            return {}

class InteractionRepository(BaseRepository):
    """Repository for interaction data operations."""
    
    def save_interaction(self, interaction_data: Dict[str, Any]) -> int:
        """Save an interaction and return interaction ID."""
        try:
            metadata_json = json.dumps(interaction_data.get('metadata', {}))
            
            query = """
                INSERT INTO interactions 
                (user_id, movie_id, interaction_type, rating, 
                 duration, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                interaction_data['user_id'],
                interaction_data['movie_id'],
                interaction_data['interaction_type'],
                interaction_data.get('rating'),
                interaction_data.get('duration'),
                interaction_data.get('timestamp', datetime.utcnow().isoformat()),
                metadata_json
            )
            
            # Execute and get the inserted ID
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                interaction_id = cursor.lastrowid
                conn.commit()
            
            logger.info(f"Saved interaction #{interaction_id}")
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error saving interaction: {str(e)}")
            return -1
    
    def get_user_interactions(self, user_id: str, limit: int = 50, 
                             offset: int = 0) -> List[Dict[str, Any]]:
        """Get interactions for a user."""
        try:
            query = """
                SELECT i.*, m.title, m.genres
                FROM interactions i
                LEFT JOIN movies m ON i.movie_id = m.movie_id
                WHERE i.user_id = ?
                ORDER BY i.timestamp DESC
                LIMIT ? OFFSET ?
            """
            
            results = self.execute_query(query, (user_id, limit, offset))
            interactions = []
            
            for row in results:
                interaction = {
                    'interaction_id': row[0],
                    'user_id': row[1],
                    'movie_id': row[2],
                    'interaction_type': row[3],
                    'rating': row[4],
                    'duration': row[5],
                    'timestamp': row[6],
                    'metadata': json.loads(row[7]) if row[7] else {},
                    'created_at': row[8],
                    'movie_title': row[9],
                    'movie_genres': json.loads(row[10]) if row[10] else []
                }
                interactions.append(interaction)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting user interactions: {str(e)}")
            return []
    
    def get_movie_interactions(self, movie_id: str, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get interactions for a movie."""
        try:
            query = """
                SELECT i.*, u.preferences
                FROM interactions i
                LEFT JOIN users u ON i.user_id = u.user_id
                WHERE i.movie_id = ?
                ORDER BY i.timestamp DESC
                LIMIT ?
            """
            
            results = self.execute_query(query, (movie_id, limit))
            interactions = []
            
            for row in results:
                interaction = {
                    'interaction_id': row[0],
                    'user_id': row[1],
                    'movie_id': row[2],
                    'interaction_type': row[3],
                    'rating': row[4],
                    'duration': row[5],
                    'timestamp': row[6],
                    'metadata': json.loads(row[7]) if row[7] else {},
                    'created_at': row[8],
                    'user_preferences': json.loads(row[9]) if row[9] else {}
                }
                interactions.append(interaction)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting movie interactions: {str(e)}")
            return []
    
    def get_recent_interactions(self, hours: int = 24, 
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent interactions."""
        try:
            query = """
                SELECT i.*, m.title, u.preferences
                FROM interactions i
                LEFT JOIN movies m ON i.movie_id = m.movie_id
                LEFT JOIN users u ON i.user_id = u.user_id
                WHERE i.timestamp >= datetime('now', ?)
                ORDER BY i.timestamp DESC
                LIMIT ?
            """
            
            time_filter = f"-{hours} hours"
            results = self.execute_query(query, (time_filter, limit))
            interactions = []
            
            for row in results:
                interaction = {
                    'interaction_id': row[0],
                    'user_id': row[1],
                    'movie_id': row[2],
                    'interaction_type': row[3],
                    'rating': row[4],
                    'duration': row[5],
                    'timestamp': row[6],
                    'metadata': json.loads(row[7]) if row[7] else {},
                    'created_at': row[8],
                    'movie_title': row[9],
                    'user_preferences': json.loads(row[10]) if row[10] else {}
                }
                interactions.append(interaction)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting recent interactions: {str(e)}")
            return []
    
    def get_interaction_matrix(self) -> pd.DataFrame:
        """Get user-item interaction matrix for collaborative filtering."""
        try:
            query = """
                SELECT user_id, movie_id, AVG(rating) as avg_rating
                FROM interactions
                WHERE rating IS NOT NULL
                GROUP BY user_id, movie_id
            """
            
            results = self.execute_query(query)
            
            # Convert to DataFrame
            data = []
            for row in results:
                data.append({
                    'user_id': row[0],
                    'movie_id': row[1],
                    'rating': row[2]
                })
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Create pivot table
            interaction_matrix = df.pivot_table(
                index='user_id',
                columns='movie_id',
                values='rating',
                aggfunc='mean'
            ).fillna(0)
            
            return interaction_matrix
            
        except Exception as e:
            logger.error(f"Error getting interaction matrix: {str(e)}")
            return pd.DataFrame()

class RecommendationRepository(BaseRepository):
    """Repository for recommendation data operations."""
    
    def save_recommendation(self, recommendation_data: Dict[str, Any]) -> int:
        """Save a recommendation."""
        try:
            query = """
                INSERT INTO recommendations 
                (user_id, movie_id, algorithm, score, generated_at)
                VALUES (?, ?, ?, ?, ?)
            """
            
            params = (
                recommendation_data['user_id'],
                recommendation_data['movie_id'],
                recommendation_data['algorithm'],
                recommendation_data['score'],
                recommendation_data.get('generated_at', 
                                      datetime.utcnow().isoformat())
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                recommendation_id = cursor.lastrowid
                conn.commit()
            
            logger.info(f"Saved recommendation #{recommendation_id}")
            return recommendation_id
            
        except Exception as e:
            logger.error(f"Error saving recommendation: {str(e)}")
            return -1
    
    def mark_recommendation_served(self, recommendation_id: int) -> bool:
        """Mark a recommendation as served."""
        try:
            query = """
                UPDATE recommendations 
                SET served_at = CURRENT_TIMESTAMP
                WHERE recommendation_id = ?
            """
            
            self.execute_update(query, (recommendation_id,))
            return True
            
        except Exception as e:
            logger.error(f"Error marking recommendation served: {str(e)}")
            return False
    
    def mark_recommendation_clicked(self, recommendation_id: int) -> bool:
        """Mark a recommendation as clicked."""
        try:
            query = """
                UPDATE recommendations 
                SET clicked = 1
                WHERE recommendation_id = ?
            """
            
            self.execute_update(query, (recommendation_id,))
            return True
            
        except Exception as e:
            logger.error(f"Error marking recommendation clicked: {str(e)}")
            return False
    
    def get_user_recommendations(self, user_id: str, 
                                limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent recommendations for a user."""
        try:
            query = """
                SELECT r.*, m.title, m.genres, m.poster_url
                FROM recommendations r
                LEFT JOIN movies m ON r.movie_id = m.movie_id
                WHERE r.user_id = ?
                ORDER BY r.generated_at DESC
                LIMIT ?
            """
            
            results = self.execute_query(query, (user_id, limit))
            recommendations = []
            
            for row in results:
                recommendation = {
                    'recommendation_id': row[0],
                    'user_id': row[1],
                    'movie_id': row[2],
                    'algorithm': row[3],
                    'score': row[4],
                    'generated_at': row[5],
                    'served_at': row[6],
                    'clicked': bool(row[7]),
                    'created_at': row[8],
                    'movie_title': row[9],
                    'movie_genres': json.loads(row[10]) if row[10] else [],
                    'poster_url': row[11]
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting user recommendations: {str(e)}")
            return []
    
    def get_recommendation_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get recommendation statistics."""
        try:
            stats = {
                'total_recommendations': 0,
                'served_recommendations': 0,
                'clicked_recommendations': 0,
                'click_through_rate': 0,
                'popular_algorithms': []
            }
            
            # Get total recommendations
            query = """
                SELECT COUNT(*) 
                FROM recommendations 
                WHERE generated_at >= date('now', ?)
            """
            result = self.execute_query(query, (f"-{days} days",))
            stats['total_recommendations'] = result[0][0] if result else 0
            
            # Get served recommendations
            query = """
                SELECT COUNT(*) 
                FROM recommendations 
                WHERE served_at IS NOT NULL 
                AND generated_at >= date('now', ?)
            """
            result = self.execute_query(query, (f"-{days} days",))
            stats['served_recommendations'] = result[0][0] if result else 0
            
            # Get clicked recommendations
            query = """
                SELECT COUNT(*) 
                FROM recommendations 
                WHERE clicked = 1 
                AND generated_at >= date('now', ?)
            """
            result = self.execute_query(query, (f"-{days} days",))
            stats['clicked_recommendations'] = result[0][0] if result else 0
            
            # Calculate CTR
            if stats['served_recommendations'] > 0:
                stats['click_through_rate'] = (
                    stats['clicked_recommendations'] / 
                    stats['served_recommendations']
                )
            
            # Get popular algorithms
            query = """
                SELECT algorithm, COUNT(*) as count
                FROM recommendations
                WHERE generated_at >= date('now', ?)
                GROUP BY algorithm
                ORDER BY count DESC
                LIMIT 5
            """
            results = self.execute_query(query, (f"-{days} days",))
            
            for row in results:
                stats['popular_algorithms'].append({
                    'algorithm': row[0],
                    'count': row[1]
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting recommendation stats: {str(e)}")
            return {}
