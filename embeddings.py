import requests
from typing import List


class Embeddings:    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.cohere.ai/v1/embed"
    
    def get_embedding(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "texts": [text],
            "model": "embed-english-v3.0",
            "input_type": "search_document"
        }
        
        response = requests.post(self.api_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Cohere API error: {response.status_code} - {response.text}")
        
        data = response.json()
        return data["embeddings"][0]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "texts": texts,
            "model": "embed-english-v3.0",
            "input_type": "search_document"
        }
        
        response = requests.post(self.api_url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Cohere API error: {response.status_code} - {response.text}")
        
        data = response.json()
        return data["embeddings"]