# Workstation / DGX image. Data stays outside the image:
#   docker build -t ancdata .
#   docker run --rm -it -e ANC_DATA_ROOT=/data -v /path/to/data:/data -v $PWD/configs:/app/configs ancdata make selftest
# GPU training image: use nvcr.io/nvidia/pytorch:24.xx-py3 as the base instead and keep the rest.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 make curl unzip git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY ancdata ./ancdata
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./
ENV ANC_DATA_ROOT=/data
CMD ["make", "smoke"]
