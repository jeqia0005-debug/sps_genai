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

# Copy the application code
COPY ./app /code/app
COPY main.py /code/
COPY cifar10_cnn.pth /code/

# Command to run the application
CMD ["/code/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
