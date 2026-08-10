import io
import json
import os

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
from app.energy_model import generate_energy_image
from app.diffusion_model import generate_diffusion_image
from app.llm_model import build_qa_model
from app.rl_reward import compute_reward, format_description, parse_response

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

# ---- Assignment 5: GPT-2 question answering + RL post-training ----
RL_TRAINING_LOG = os.environ.get("RL_TRAINING_LOG", "rl_training_log.json")
LLM_DEVICE = os.environ.get("LLM_DEVICE", "cpu")

# GPT-2 takes a few seconds to load, so each stage is built on first use and
# then kept around. Loading at import time would make the whole API unavailable
# whenever an adapter is missing.
_qa_models = {}


def get_qa_model(stage: str):
    """Return the cached question-answering model for a pipeline stage."""
    if stage not in _qa_models:
        try:
            _qa_models[stage] = build_qa_model(stage=stage, device=LLM_DEVICE)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error))
    return _qa_models[stage]


def answer_with(stage: str, request: "QuestionRequest") -> dict:
    """Run one stage of the pipeline and score what it produced."""
    model = get_qa_model(stage)
    output = model.answer(
        request.question,
        request.context,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )
    parsed = parse_response(output.text)
    result = {
        "stage": stage,
        "completion": output.text,
        "reasoning": parsed.reasoning,
        "answer": parsed.answer,
        "format_compliant": parsed.well_formed,
        "finished": output.finished,
    }
    if request.gold_answer:
        breakdown = compute_reward(output.text, request.gold_answer, finished=output.finished)
        result["reward"] = breakdown.as_dict()
    return result
# -------------------------------------------------------------------


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


class QuestionRequest(BaseModel):
    """A SQuAD-style question: the model answers using the given context."""

    question: str
    context: str
    gold_answer: str | None = None  # optional, enables reward scoring
    max_new_tokens: int = 64
    temperature: float = 0.7


class RewardRequest(BaseModel):
    """Score an arbitrary completion with the reinforcement learning reward."""

    completion: str
    gold_answer: str = ""
    finished: bool = True


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

@app.get("/generate-energy")
def generate_energy(num_images: int = 4, steps: int = 256):
    """Generate CIFAR-10-like images with the Energy-Based Model via Langevin dynamics."""
    if num_images < 1 or num_images > 16:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 16.")
    if steps < 1 or steps > 1000:
        raise HTTPException(status_code=400, detail="steps must be between 1 and 1000.")

    buffer = generate_energy_image(
        weights_path="cifar10_energy.pth",
        num_images=num_images,
        steps=steps,
    )
    return StreamingResponse(buffer, media_type="image/png")


@app.get("/generate-diffusion")
def generate_diffusion(num_images: int = 4, diffusion_steps: int = 20):
    """Generate CIFAR-10-like images with the diffusion model via reverse diffusion."""
    if num_images < 1 or num_images > 16:
        raise HTTPException(status_code=400, detail="num_images must be between 1 and 16.")
    if diffusion_steps < 1 or diffusion_steps > 100:
        raise HTTPException(status_code=400, detail="diffusion_steps must be between 1 and 100.")

    buffer = generate_diffusion_image(
        weights_path="cifar10_diffusion.pth",
        num_images=num_images,
        diffusion_steps=diffusion_steps,
    )
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/generate_with_llm")
def generate_with_llm(request: QuestionRequest):
    """Answer a question with GPT-2 fine-tuned on SQuAD (Module 8 activity)."""
    return answer_with("sft", request)


@app.post("/generate_with_rl")
def generate_with_rl(request: QuestionRequest):
    """Answer with the RL post-trained model, which uses the required format."""
    return answer_with("rl", request)


@app.post("/rl/compare")
def compare_llm_and_rl(request: QuestionRequest):
    """Run both stages on the same question to show what the RL stage changed."""
    return {
        "question": request.question,
        "target_format": format_description()["template"],
        "supervised": answer_with("sft", request),
        "reinforcement_learned": answer_with("rl", request),
    }


@app.post("/rl/reward")
def score_completion(request: RewardRequest):
    """Score a completion with the reward function used during RL training.

    This runs the algorithm itself, without loading a model, so the reward
    shaping can be inspected independently of the policy.
    """
    breakdown = compute_reward(
        request.completion, request.gold_answer, finished=request.finished
    )
    return {"completion": request.completion, **breakdown.as_dict()}


@app.get("/rl/format")
def get_target_format():
    """Describe the response format the RL stage was rewarded for producing."""
    return format_description()


@app.get("/rl/training-curve")
def get_training_curve():
    """Return the return curve and before/after evaluation of the RL stage."""
    if not os.path.exists(RL_TRAINING_LOG):
        raise HTTPException(
            status_code=503,
            detail=f"Training log '{RL_TRAINING_LOG}' not found. Run train_rl.py first.",
        )

    with open(RL_TRAINING_LOG, encoding="utf-8") as handle:
        log = json.load(handle)

    return {
        "config": log.get("config", {}),
        "duration_seconds": log.get("duration_seconds", 0.0),
        "curve": [
            {
                "epoch": epoch["epoch"],
                "mean_return": round(epoch["mean_return"], 4),
                "format_rate": round(epoch["format_rate"], 4),
                "answer_f1": round(epoch["answer_f1"], 4),
            }
            for epoch in log.get("epochs", [])
        ],
        "samples": log.get("samples", []),
    }