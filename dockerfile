FROM python:3.9.22-slim-bullseye

RUN apt-get update && \
apt-get install -y \
ninja-build \
cmake \
clang \
build-essential && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

RUN cmake --version

RUN clang --version

RUN mkdir /app/

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/BitNet

RUN pip install --no-cache-dir -r requirements.txt

RUN python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

RUN mkdir /app/models

RUN mv /app/BitNet/build/bin /app/libs

RUN mv /app/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf /app/models/ggml-model-i2_s.gguf

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]