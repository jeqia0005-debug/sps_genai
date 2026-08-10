# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock /code/

# Install dependencies using uv
RUN uv sync --frozen

# --- Assignment 2: install PyTorch (CPU) + image libs into the uv venv ---
ENV VIRTUAL_ENV=/code/.venv
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install pillow
# ------------------------------------------------------------------------

# --- Assignment 5: transformers + the pretrained GPT-2 weights ---
# The weights are baked into the image so the container needs no network at
# runtime; only the LoRA adapters trained by this repository are copied in.
ENV HF_HOME=/code/.cache/huggingface
RUN uv pip install transformers
RUN /code/.venv/bin/python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoModelForCausalLM.from_pretrained('openai-community/gpt2'); \
AutoTokenizer.from_pretrained('openai-community/gpt2')"
# -----------------------------------------------------------------

# Copy the application code
COPY ./app /code/app
COPY main.py /code/
COPY cifar10_cnn.pth /code/
COPY mnist_gan_generator.pth /code/
COPY cifar10_energy.pth /code/
COPY cifar10_diffusion.pth /code/
COPY gpt2_squad_lora.pt /code/
COPY gpt2_rl_lora.pt /code/
COPY rl_training_log.json /code/

# Command to run the application
CMD ["/code/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
