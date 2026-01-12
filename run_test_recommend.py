from main import MovieRecommendationSystem
from fastapi.testclient import TestClient
import json

def test_recommend(user_id='user1'):
    system = MovieRecommendationSystem()
    app = system.create_api_app()
    client = TestClient(app)

    url = f'/api/v1/recommendations/{user_id}?n_recommendations=5'
    resp = client.get(url)
    print('status_code:', resp.status_code)
    try:
        data = resp.json()
        print('response:', json.dumps(data, indent=2))
    except Exception as e:
        print('Error parsing response:', e)

if __name__ == '__main__':
    test_recommend()
