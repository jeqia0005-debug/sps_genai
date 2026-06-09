from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

app = FastAPI(title="SPS GenAI API")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    vector = embedding_model.calculate_embedding(request.word)
    if not vector.any():
        raise HTTPException(status_code=404, detail=f"No embedding available for '{request.word}'.")
    return {
        "word": request.word,
        "model": embedding_model.model_name,
        "dimension": len(vector),
        "embedding": vector.tolist(),
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    similarity = embedding_model.calculate_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": similarity}