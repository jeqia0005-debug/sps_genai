import io

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import torch
import torchvision.transforms as transforms
from PIL import Image

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.cnn_model import CNN
from app.gan_model import generate_digit_image

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

# ---- Assignment 2: image classifier ----
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

cnn_model = CNN(num_classes=10)
cnn_model.load_state_dict(torch.load("cifar10_cnn.pth", map_location="cpu"))
cnn_model.eval()  # inference mode

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
# -----------------------------------------


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


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Accept an uploaded image and return the predicted CIFAR10 class."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = image_transform(image).unsqueeze(0)  # -> [1, 3, 64, 64]

    with torch.no_grad():
        outputs = cnn_model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        idx = int(probs.argmax())

    return {
        "prediction": CIFAR10_CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
    }

@app.get("/generate-digit")
def generate_digit(num_images: int = 1):
    """Generate handwritten digit image(s) using the trained GAN generator."""
    if num_images < 1 or num_images > 16:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 16.")

    buffer = generate_digit_image(weights_path="mnist_gan_generator.pth", num_images=num_images)
    return StreamingResponse(buffer, media_type="image/png")