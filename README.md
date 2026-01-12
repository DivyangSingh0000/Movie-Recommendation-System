# Movie Recommendation System

A comprehensive movie recommendation system implementing multiple algorithms including collaborative filtering, content-based filtering, and hybrid approaches.

## Features

- **Multiple Recommendation Algorithms**:
  - User-based Collaborative Filtering
  - Item-based Collaborative Filtering
  - Matrix Factorization (SVD)
  - Content-based Filtering
  - Hybrid Recommendation System

- **Data Pipeline**:
  - Automated data validation and preprocessing
  - Feature engineering and transformation
  - Data storage and management

- **RESTful API**:
  - FastAPI-based REST endpoints
  - Comprehensive documentation (Swagger/ReDoc)
  - Authentication and rate limiting

- **Monitoring & Logging**:
  - Comprehensive logging system
  - Performance metrics
  - Error tracking

## Project Structure

```
movie-recommendation-system/
├── automation/               # Data validation automation
│   ├── src/
│   │   ├── main.py           # CLI entrypoint
│   │   └── automation.py     # Core validation logic
│   └── tests/
├── data_pipeline/            # ETL and data processing
│   ├── extract/              # Data extraction
│   ├── transform/            # Data transformation
│   └── load/                 # Data loading
├── recommendation/           # Recommendation algorithms
│   ├── collaborative/        # Collaborative filtering
│   ├── content_based/        # Content-based filtering
│   ├── hybrid/               # Hybrid approaches
│   └── models/               # Trained model storage
├── api/                      # REST API
│   ├── endpoints/            # API endpoints
│   ├── middleware/           # API middleware
│   └── schemas/              # Pydantic schemas
├── storage/                  # Database interactions
│   ├── repositories/         # Data repositories
│   └── migrations/           # Database migrations
├── monitoring/               # Logging and monitoring
│   ├── logging/              # Logging configuration
│   ├── metrics/              # Performance metrics
│   └── alerts/               # Alert system
├── main.py                   # Main application
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── run.sh                    # Runner script
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd movie-recommendation-system
```

2. Run the setup script:

```bash
chmod +x run.sh
./run.sh setup
```

This will:

- Create a Python virtual environment
- Install all dependencies
- Create a sample data structure under `data/input`

## Usage

1. Validate Input Data

```bash
./run.sh validate
```

2. Process Data

```bash
./run.sh process
```

3. Train Models

```bash
./run.sh train
```

4. Start API Server

```bash
./run.sh serve
```

- **Working API URL:** http://localhost:8000
- **API Documentation (Swagger UI):** http://localhost:8000/docs
- **API Documentation (ReDoc):** http://localhost:8000/redoc

- **Fetch OpenAPI (JSON) / Documentation programmatically:**

```bash
curl -sS http://localhost:8000/openapi.json -o openapi.json
```

You can open `openapi.json` in tooling that consumes OpenAPI/Swagger definitions or publish it as needed.

5. Get Recommendations

Using the script:

```bash
./run.sh recommend user1
```

Using the API (example):

```bash
curl -X GET "http://localhost:8000/api/v1/recommendations/user1?n_recommendations=10"
```

## API Endpoints (summary)

- Recommendations
  - `GET /api/v1/recommendations/{user_id}` - Get recommendations for a user
  - `GET /api/v1/recommendations/similar/{movie_id}` - Get similar movies
  - `POST /api/v1/recommendations/batch` - Get batch recommendations
  - `POST /api/v1/interactions` - Record user interaction

- Movies
  - `GET /api/v1/movies/{movie_id}` - Get movie details
  - `GET /api/v1/movies` - Search movies
  - `GET /api/v1/movies/popular` - Get popular movies
  - `GET /api/v1/genres` - Get all genres

- Users
  - `GET /api/v1/users/{user_id}/profile` - Get user profile
  - `GET /api/v1/users/{user_id}/history` - Get user history
  - `PUT /api/v1/users/{user_id}/preferences` - Update user preferences
  - `DELETE /api/v1/users/{user_id}` - Delete user data

## Data Format

### Input Data Structure

```
input_folder/
├── metadata.json      # Movie metadata
├── data.csv           # User interactions
└── assets/            # Movie assets (posters, etc.)
```

### `metadata.json` format (example)

```json
{
  "movies": [
    {
      "movie_id": "tt1234567",
      "title": "Movie Title",
      "genres": ["Action", "Adventure"],
      "release_year": 2023,
      "description": "Movie description",
      "directors": ["Director Name"],
      "actors": ["Actor 1", "Actor 2"],
      "imdb_rating": 7.5,
      "duration_minutes": 120,
      "language": "English",
      "country": "USA",
      "poster_url": "http://example.com/poster.jpg",
      "trailer_url": "http://example.com/trailer.mp4"
    }
  ]
}
```

### `data.csv` format (example)

```csv
user_id,movie_id,rating,timestamp,interaction_type
user1,tt1234567,4.5,2024-01-15T10:30:00,rating
user2,tt1234567,5.0,2024-01-14T20:15:00,rating
```

## Algorithm Details

### Collaborative Filtering

- User-Based: Finds users with similar tastes and recommends movies they liked
- Item-Based: Finds similar movies based on user ratings
- Matrix Factorization: Decomposes user-item matrix into latent factors

### Content-Based Filtering

- Uses movie metadata (genres, description, actors, etc.)
- Computes similarity based on feature vectors
- TF-IDF for text features

### Hybrid Approach

- Combines multiple algorithms with weighted scores
- Dynamic algorithm selection based on user context
- Cascade refinement of recommendations

## Configuration

### Environment variables

```bash
# Database
DATABASE_URL=sqlite:///data/recommendation.db

# API
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your_api_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/recommendation_system.log
```

### Configuration file (optional)

Create `config.yaml` in the project root:

```yaml
database:
  type: sqlite
  path: data/recommendation.db

api:
  host: 0.0.0.0
  port: 8000
  api_keys:
    - default_key: test_key_123

recommendation:
  default_algorithm: hybrid
  n_recommendations: 10
  feature_weights:
    genres: 0.4
    description: 0.3
    directors: 0.1
    actors: 0.1
    year: 0.05
    rating: 0.05
```

## Monitoring

- Logs
  - Application logs: `logs/recommendation_system.log`
  - API access logs: `logs/api_access.log`
  - Error logs: `logs/error.log`

- Metrics
  - Recommendation counts and CTR
  - Algorithm performance metrics
  - System resource usage

## Testing

Run all tests:

```bash
./run.sh test
```

Run specific test categories:

```bash
# Automation tests
pytest automation/tests/ -v

# Data pipeline tests
pytest data_pipeline/tests/ -v

# Recommendation tests
pytest recommendation/tests/ -v

# API tests
pytest api/tests/ -v
```

## Deployment

### Docker

```bash
# Build the Docker image
docker build -t movie-recommendation .

# Run the container
docker run -p 8000:8000 movie-recommendation
```

### Docker Compose

```bash
docker-compose up -d
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Performance Optimization

- Caching: Redis for caching frequent recommendations and DB query caching
- Model prediction caching
- Scaling: Horizontal scaling of API servers, DB read replicas, message queue for async processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Support

For support, please:

- Check the documentation
- Search existing issues
- Create a new issue with detailed information

## Acknowledgments

- The MovieLens dataset team for inspiration
- Scikit-learn and FastAPI communities
- All contributors to open-source machine learning libraries

---

This repository implements a modular, extensible movie recommendation system with automation for data validation, an ETL pipeline, multiple recommendation algorithms, a FastAPI service layer, and storage/repository patterns for production readiness.
Movie Recommendation System

Repository layout:

- `automation/` - Data validation automation (CLI) — includes examples and tests
- `data_pipeline/` - ETL stages: `extract/`, `transform/`, `load/`
- `recommendation/` - Recommendation algorithms (collaborative, content-based, hybrid)
- `api/` - FastAPI app skeleton
- `storage/` - simple SQLite repository helpers
- `monitoring/` - logging utilities

Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the validation automation with example input:

```bash
python automation/src/main.py automation/examples/input_valid -o automation/examples/output_valid
```

Run the API (dev):

```bash
uvicorn api.app:app --reload
```
