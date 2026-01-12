from main import MovieRecommendationSystem
from fastapi.testclient import TestClient
import json

def test_openapi():
    system = MovieRecommendationSystem()
    app = system.create_api_app()
    client = TestClient(app)

    resp = client.get('/openapi.json')
    print('status_code:', resp.status_code)
    if resp.status_code == 200:
        # print a short summary
        data = resp.json()
        print('title:', data.get('info', {}).get('title'))
        print('paths_count:', len(data.get('paths', {})))
        # save to file
        with open('openapi.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print('Saved openapi.json')
    else:
        print('Response text:', resp.text)

if __name__ == '__main__':
    test_openapi()
