from services.embeddings import Embeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
embedding_model = Embeddings(api_key=api_key)


text = "HELLO WORLD, WHAT IS UP?"
print(embedding_model.get_embedding(text))