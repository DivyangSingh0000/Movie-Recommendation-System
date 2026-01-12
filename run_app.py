from main import MovieRecommendationSystem
import uvicorn

def run():
    system = MovieRecommendationSystem()
    app = system.create_api_app()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")

if __name__ == "__main__":
    run()
