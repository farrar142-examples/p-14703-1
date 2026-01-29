FROM python:3.11-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir \
    huggingface_hub \
    transformers \
    sentencepiece \
    protobuf \
    gguf \
    safetensors

# PyTorch CPU 버전 별도 설치
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# llama.cpp 클론 및 CMake 빌드
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /app/llama.cpp \
    && cd /app/llama.cpp \
    && cmake -B build \
    && cmake --build build --target llama-quantize -j$(nproc) \
    && cp build/bin/llama-quantize /app/llama.cpp/

WORKDIR /app

# 변환 스크립트 복사
COPY converter.py /app/converter.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# models 디렉토리 생성
RUN mkdir -p /models/gguf /models/hf_cache

# 환경변수
ENV MODELS_DIR=/models
ENV HF_HOME=/models/hf_cache

ENTRYPOINT ["/app/entrypoint.sh"]
