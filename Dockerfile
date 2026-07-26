FROM python:3.12-slim

# Run as a non-root user rather than the image's default root.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app
# WORKDIR is created by root at this point in the build — hand it to `user`
# now, or ingest.py (running as `user` below) can't mkdir chroma-store/ in it.
RUN chown user:user /home/user/app

# deps first for layer caching
COPY --chown=user requirements.txt .
# CPU-only torch first — otherwise pip pulls the default CUDA build (multiple
# GB of nvidia-* packages) for a target with no GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

USER user
COPY --chown=user . .

# Build the vector store + cache the embedding model into the image (no API key needed).
RUN python ingest.py

# Model is already cached from the step above — skip runtime freshness-check
# calls to huggingface.co on every container start (pure latency, no benefit).
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
# Shell form so $PORT expands (Cloud Run injects 8080); exec keeps uvicorn as
# PID 1 so it receives SIGTERM on scale-down. :-7860 default keeps local
# `docker run -p 7860:7860` working unchanged.
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
