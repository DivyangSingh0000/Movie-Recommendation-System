#!/bin/bash

# Movie Recommendation System Runner Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
LOG_DIR="${PROJECT_DIR}/logs"
DATA_DIR="${PROJECT_DIR}/data"
INPUT_DIR="${DATA_DIR}/input"
OUTPUT_DIR="${DATA_DIR}/output"
MODELS_DIR="${PROJECT_DIR}/models"

# Create necessary directories
mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${INPUT_DIR}" "${OUTPUT_DIR}" "${MODELS_DIR}"

print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        Movie Recommendation System                           ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${YELLOW}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    print_step "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 1 ]]; then
            print_success "Python $PYTHON_VERSION is installed"
        else
            print_error "Python 3.8 or higher is required (found $PYTHON_VERSION)"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check for pip
    if command -v pip3 &>/dev/null; then
        print_success "pip3 is installed"
    else
        print_error "pip3 is not installed"
        exit 1
    fi
}

setup_venv() {
    print_step "Setting up Python virtual environment..."
    
    if [[ ! -d "${VENV_DIR}" ]]; then
        python3 -m venv "${VENV_DIR}"
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source "${VENV_DIR}/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_success "Virtual environment activated"
}

install_dependencies() {
    print_step "Installing Python dependencies..."
    
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

validate_data() {
    print_step "Validating input data..."
    
    if [[ ! -d "${INPUT_DIR}" ]]; then
        print_error "Input directory not found: ${INPUT_DIR}"
        print_step "Creating sample input structure..."
        create_sample_data
    fi
    
    # Run the validation automation
    if python -m automation.src.main "${INPUT_DIR}" -o "${OUTPUT_DIR}"; then
        print_success "Data validation passed"
        return 0
    else
        print_error "Data validation failed"
        echo "Check ${OUTPUT_DIR}/error_summary.json for details"
        return 1
    fi
}

create_sample_data() {
    print_step "Creating sample data structure..."
    
    # Create metadata.json
    cat > "${INPUT_DIR}/metadata.json" << EOF
{
  "movies": [
    {
      "movie_id": "tt0111161",
      "title": "The Shawshank Redemption",
      "genres": ["Drama"],
      "release_year": 1994,
      "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
      "directors": ["Frank Darabont"],
      "actors": ["Tim Robbins", "Morgan Freeman", "Bob Gunton"],
      "imdb_rating": 9.3,
      "duration_minutes": 142,
      "language": "English",
      "country": "USA"
    },
    {
      "movie_id": "tt0068646",
      "title": "The Godfather",
      "genres": ["Crime", "Drama"],
      "release_year": 1972,
      "description": "The aging patriarch of an organized crime dynasty transfers control to his reluctant son.",
      "directors": ["Francis Ford Coppola"],
      "actors": ["Marlon Brando", "Al Pacino", "James Caan"],
      "imdb_rating": 9.2,
      "duration_minutes": 175,
      "language": "English",
      "country": "USA"
    },
    {
      "movie_id": "tt0071562",
      "title": "The Godfather Part II",
      "genres": ["Crime", "Drama"],
      "release_year": 1974,
      "description": "The early life and career of Vito Corleone in 1920s New York is portrayed while his son, Michael, expands and tightens his grip on the family crime syndicate.",
      "directors": ["Francis Ford Coppola"],
      "actors": ["Al Pacino", "Robert De Niro", "Robert Duvall"],
      "imdb_rating": 9.0,
      "duration_minutes": 202,
      "language": "English",
      "country": "USA"
    }
  ]
}
EOF
    
    # Create data.csv
    cat > "${INPUT_DIR}/data.csv" << EOF
user_id,movie_id,rating,timestamp,interaction_type
user1,tt0111161,5.0,2024-01-15T10:30:00,rating
user1,tt0068646,4.5,2024-01-14T20:15:00,rating
user2,tt0111161,4.0,2024-01-13T15:45:00,rating
user2,tt0071562,4.8,2024-01-12T19:20:00,rating
user3,tt0068646,5.0,2024-01-11T21:00:00,rating
user3,tt0071562,4.2,2024-01-10T18:30:00,rating
EOF
    
    # Create assets directory with sample files
    mkdir -p "${INPUT_DIR}/assets/posters"
    
    # Create placeholder files
    touch "${INPUT_DIR}/assets/posters/tt0111161.jpg"
    touch "${INPUT_DIR}/assets/posters/tt0068646.jpg"
    touch "${INPUT_DIR}/assets/posters/tt0071562.jpg"
    
    print_success "Sample data created in ${INPUT_DIR}"
}

load_and_process_data() {
    print_step "Loading and processing data..."
    
    if python -c "import main; system = main.MovieRecommendationSystem(); system.load_data('${INPUT_DIR}')"; then
        print_success "Data loaded and processed successfully"
        return 0
    else
        print_error "Failed to load and process data"
        return 1
    fi
}

train_models() {
    print_step "Training recommendation models..."
    
    if python -c "import main; system = main.MovieRecommendationSystem(); system.train_models()"; then
        print_success "Models trained successfully"
        return 0
    else
        print_error "Failed to train models"
        return 1
    fi
}

start_api() {
    print_step "Starting API server..."
    
    local host="${1:-0.0.0.0}"
    local port="${2:-8000}"
    
    echo -e "${YELLOW}"
    echo "API Documentation will be available at:"
    echo "  - Swagger UI: http://${host}:${port}/docs"
    echo "  - ReDoc: http://${host}:${port}/redoc"
    echo -e "${NC}"
    
    # Run the API server
    uvicorn "main:MovieRecommendationSystem().create_api_app()" --host "$host" --port "$port" --reload
}

run_recommendations() {
    print_step "Running recommendations..."
    
    local user_id="${1:-user1}"
    local count="${2:-5}"
    local algorithm="${3:-hybrid}"
    
    python -c "
import main
system = main.MovieRecommendationSystem()
recommendations = system.get_recommendations('$user_id', $count, '$algorithm')
print(f'\nRecommendations for {user_id} (using {algorithm}):')
print('=' * 60)
for i, rec in enumerate(recommendations, 1):
    print(f'{i:2}. {rec["title"]} ({rec.get("release_year", "N/A")})')
    print(f'    Score: {rec["score"]:.3f}')
    print(f'    Genres: {", ".join(rec.get("genres", []))}')
    print()
"
}

show_help() {
    print_header
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup               Setup the project environment"
    echo "  validate            Validate input data"
    echo "  process             Load and process data"
    echo "  train               Train recommendation models"
    echo "  serve [host] [port] Start the API server"
    echo "  recommend [user]    Get recommendations for a user"
    echo "  test                Run all tests"
    echo "  clean               Clean generated files"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup            # Setup the project"
    echo "  $0 serve            # Start API on 0.0.0.0:8000"
    echo "  $0 serve localhost 8080  # Start API on localhost:8080"
    echo "  $0 recommend user1  # Get recommendations for user1"
    echo ""
}

run_tests() {
    print_step "Running tests..."
    
    # Run automation tests
    if [[ -d "automation/tests" ]]; then
        python -m pytest automation/tests/ -v
    fi
    
    # Run data pipeline tests
    if [[ -d "data_pipeline/tests" ]]; then
        python -m pytest data_pipeline/tests/ -v
    fi
    
    # Run recommendation tests
    if [[ -d "recommendation/tests" ]]; then
        python -m pytest recommendation/tests/ -v
    fi
    
    print_success "Tests completed"
}

clean_project() {
    print_step "Cleaning project..."
    
    # Remove virtual environment
    if [[ -d "${VENV_DIR}" ]]; then
        rm -rf "${VENV_DIR}"
        print_success "Virtual environment removed"
    fi
    
    # Remove logs
    if [[ -d "${LOG_DIR}" ]]; then
        rm -rf "${LOG_DIR}"
        print_success "Logs removed"
    fi
    
    # Remove processed data
    if [[ -d "${DATA_DIR}/processed" ]]; then
        rm -rf "${DATA_DIR}/processed"
        print_success "Processed data removed"
    fi
    
    # Remove models
    if [[ -d "${MODELS_DIR}" ]]; then
        rm -rf "${MODELS_DIR}"
        print_success "Models removed"
    fi
    
    # Remove output
    if [[ -d "${OUTPUT_DIR}" ]]; then
        rm -rf "${OUTPUT_DIR}"
        print_success "Output files removed"
    fi
    
    # Remove Python cache files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type f -name ".coverage" -delete 2>/dev/null || true
    
    print_success "Project cleaned"
}

# Main script execution
main() {
    print_header
    
    case "${1:-help}" in
        setup)
            check_requirements
            setup_venv
            install_dependencies
            create_sample_data
            ;;
        
        validate)
            setup_venv
            validate_data
            ;;
        
        process)
            setup_venv
            validate_data && load_and_process_data
            ;;
        
        train)
            setup_venv
            train_models
            ;;
        
        serve)
            setup_venv
            local host="${2:-0.0.0.0}"
            local port="${3:-8000}"
            start_api "$host" "$port"
            ;;
        
        recommend)
            setup_venv
            local user="${2:-user1}"
            local count="${3:-5}"
            local algorithm="${4:-hybrid}"
            run_recommendations "$user" "$count" "$algorithm"
            ;;
        
        test)
            setup_venv
            run_tests
            ;;
        
        clean)
            clean_project
            ;;
        
        help|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
